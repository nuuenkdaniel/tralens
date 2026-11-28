package app.tralens.reader;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import android.webkit.CookieManager;
import androidx.core.app.NotificationCompat;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class OfflineDownloadService extends Service {
    private static final int NOTICE = 702;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    static volatile boolean running;
    private volatile boolean stopped;
    // Only a connection belonging to the current attempt can be cancelled.
    private static String attempt;
    private static HttpURLConnection connection;

    static synchronized void interrupt(String token) {
        if (token.equals(attempt) && connection != null) connection.disconnect();
    }

    @Override public IBinder onBind(Intent intent) { return null; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        running = true;
        if (Build.VERSION.SDK_INT >= 26) getSystemService(NotificationManager.class).createNotificationChannel(
            new NotificationChannel("offline", "Offline downloads", NotificationManager.IMPORTANCE_LOW));
        startForeground(NOTICE, notification("Preparing download", 0, 0));
        worker.execute(() -> {
            try {
                JSONObject job;
                while (!stopped && (job = next()) != null) download(job);
            } catch (Exception ignored) {
                // A persisted manifest can be resumed when the reader reopens.
            } finally { stopSelfResult(startId); }
        });
        return START_STICKY;
    }

    private Notification notification(String label, int saved, int total) {
        Intent open = new Intent(this, MainActivity.class).setAction("app.tralens.reader.DOWNLOADS")
            .setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        return new NotificationCompat.Builder(this, "offline").setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Saving for offline reading").setContentText(label).setOnlyAlertOnce(true)
            .setProgress(total, saved, total == 0).setOngoing(true)
            .setContentIntent(PendingIntent.getActivity(this, NOTICE, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT))
            .build();
    }

    private JSONObject next() throws Exception {
        synchronized (ReaderDownloads.LOCK) {
            JSONObject first = null;
            for (String key : ReaderDownloads.store(this).getAll().keySet()) {
                JSONObject job = ReaderDownloads.lookup(this, key);
                if ("queued".equals(job.getString("state")) && UploadQueue.origin(this).equals(job.getString("origin"))
                    && (first == null || first.getLong("created") > job.getLong("created"))) first = job;
            }
            return first;
        }
    }

    private boolean current(JSONObject job) throws Exception {
        JSONObject live = ReaderDownloads.lookup(this, job.getString("generation"));
        return !stopped && live != null && "queued".equals(live.getString("state"))
            && job.getString("attempt").equals(live.getString("attempt"));
    }

    private HttpURLConnection open(JSONObject job, String path, String cookie) throws Exception {
        synchronized (ReaderDownloads.LOCK) {
            if (!current(job) || !job.getString("origin").equals(UploadQueue.origin(this))) throw new IOException();
            synchronized (OfflineDownloadService.class) {
                attempt = job.getString("attempt");
                connection = UploadService.authenticatedConnection(job.getString("origin") + path, cookie);
                return connection;
            }
        }
    }

    private void download(JSONObject job) throws Exception {
        String key = job.getString("generation");
        File dir = ReaderDownloads.directory(this, key);
        File temp = new File(dir, "partial");
        try {
            String cookie = CookieManager.getInstance().getCookie(job.getString("origin") + "/api/library");
            if (cookie == null) throw new IOException("Sign in to resume this download.");
            HttpURLConnection config = open(job, "/api/config", cookie);
            try {
                if (!job.getString("user").equals(new JSONObject(UploadService.response(config)).optString("username")))
                    throw new IOException("Sign in to the account that started this download.");
            } finally { config.disconnect(); }
            JSONArray urls = job.getJSONArray("urls");
            for (int i = job.getInt("saved"); i < urls.length(); i++) {
                getSystemService(NotificationManager.class).notify(NOTICE,
                    notification(job.getString("title") + " · " + (i + 1) + " of " + urls.length(), i, urls.length()));
                HttpURLConnection request = open(job, urls.getString(i), cookie);
                try {
                    int status = request.getResponseCode();
                    if (status == 401) throw new IOException("Sign in to resume this download.");
                    if (status != 200) throw new IOException("Page unavailable. Reconnect and resume the download.");
                    String type = request.getContentType();
                    String ext = "image/jpeg".equals(type) ? ".jpg" : "image/png".equals(type) ? ".png" :
                        "image/webp".equals(type) ? ".webp" : null;
                    if (ext == null) throw new IOException("Server did not return a page image.");
                    long expected = request.getContentLength(), size = 0;
                    try (InputStream input = request.getInputStream(); FileOutputStream output = new FileOutputStream(temp)) {
                        byte[] bytes = new byte[65536];
                        int n;
                        while ((n = input.read(bytes)) != -1) { output.write(bytes, 0, n); size += n; }
                    }
                    if (size == 0 || (expected >= 0 && size != expected)) throw new IOException("Incomplete page. Tap Resume.");
                    synchronized (ReaderDownloads.LOCK) {
                        if (!current(job)) return;
                        if (!temp.renameTo(new File(dir, i + ext))) throw new IOException("Could not save the page. Check device storage.");
                        job.put("saved", i + 1);
                        if (i + 1 == urls.length()) job.put("state", "complete");
                        ReaderDownloads.write(this, job);
                    }
                } finally { request.disconnect(); }
            }
        } catch (Exception e) {
            synchronized (ReaderDownloads.LOCK) {
                if (current(job)) {
                    // Read the last committed count, including after an interrupted page write.
                    JSONObject live = ReaderDownloads.lookup(this, key);
                    String detail = dir.getUsableSpace() < 1024 * 1024 ? "Device storage is full." :
                        e.getMessage() != null && (e.getMessage().startsWith("Sign in") || e.getMessage().startsWith("Page unavailable"))
                            ? e.getMessage() : "Download interrupted. Reconnect and tap Resume.";
                    ReaderDownloads.write(this, live.put("state", "paused").put("error", detail));
                }
            }
        } finally {
            temp.delete();
            synchronized (OfflineDownloadService.class) {
                if (job.getString("attempt").equals(attempt)) { connection = null; attempt = null; }
            }
        }
    }

    @Override public void onTimeout(int id, int type) { stopped = true; stopSelf(); }
    @Override public void onDestroy() {
        stopped = true;
        running = false;
        synchronized (OfflineDownloadService.class) { if (connection != null) connection.disconnect(); }
        worker.shutdownNow();
        super.onDestroy();
    }
}
