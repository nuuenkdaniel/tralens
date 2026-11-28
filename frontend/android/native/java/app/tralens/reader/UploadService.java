package app.tralens.reader;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.IBinder;
import android.webkit.CookieManager;
import androidx.core.app.NotificationCompat;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class UploadService extends Service {
    static volatile String activeId;
    static volatile int percent;
    private static volatile UploadService instance;
    private volatile String cancelled;

    static boolean running() { return instance != null; }

    static void cancel(String id) {
        UploadService service = instance;
        if (service != null) service.cancelCurrent(id);
    }

    private synchronized void cancelCurrent(String id) {
        if (id.equals(activeId)) {
            cancelled = id;
            closeConnection();
        }
    }
    private static final int NOTICE = 701;
    private static final long LIMIT = 4L << 30;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private volatile HttpURLConnection connection;
    private volatile boolean stopped;

    @Override public IBinder onBind(Intent intent) { return null; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        instance = this;
        if (Build.VERSION.SDK_INT >= 26) getSystemService(NotificationManager.class).createNotificationChannel(
            new NotificationChannel("uploads", "Manga uploads", NotificationManager.IMPORTANCE_LOW));
        startForeground(NOTICE, notification("Preparing upload", -1));
        worker.execute(() -> {
            try {
                JSONObject job;
                while (!stopped && (job = next()) != null) send(job);
            } catch (Exception ignored) {
                // A manifest still on disk can be resumed from the reader.
            } finally { stopSelfResult(startId); }
        });
        return START_STICKY;
    }

    private JSONObject next() throws Exception {
        JSONArray jobs = UploadQueue.jobs(this);
        for (int i = 0; i < jobs.length(); i++) {
            JSONObject item = jobs.getJSONObject(i);
            if ("queued".equals(item.getString("state")) && UploadQueue.origin(this).equals(item.getString("origin"))) return item;
        }
        return null;
    }

    private Notification notification(String text, int progress) {
        Intent open = new Intent(this, MainActivity.class).setAction("app.tralens.reader.QUEUE")
            .setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        return new NotificationCompat.Builder(this, "uploads").setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("Tralens uploads").setContentText(text).setOnlyAlertOnce(true).setOngoing(true)
            .setContentIntent(PendingIntent.getActivity(this, NOTICE, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT))
            .setProgress(100, Math.max(0, progress), progress < 0).build();
    }

    private static final class RetryableResponse extends IOException {
        RetryableResponse(String message) { super(message); }
    }

    private void send(JSONObject job) throws Exception {
        String id = job.getString("id");
        percent = -1;
        activeId = id;
        try {
            JSONArray live = UploadQueue.jobs(this);
            boolean present = false;
            for (int i = 0; i < live.length(); i++) present |= id.equals(live.getJSONObject(i).getString("id"));
            if (!present) return;
            for (int attempt = 0; ; attempt++) {
                try { transfer(job, id); break; }
                catch (java.net.SocketException | java.net.SocketTimeoutException | java.net.UnknownHostException | RetryableResponse e) {
                    if (attempt >= 2 || stopped || id.equals(cancelled)) throw e;
                    closeConnection();
                    percent = -1;
                    Thread.sleep(attempt == 0 ? 3000 : 10000);
                }
            }
            UploadQueue.update(this, id, null, "");
        } catch (Exception e) {
            String message = e instanceof IOException ? e.getMessage() : null;
            if (message == null || e instanceof java.net.SocketException || e instanceof java.net.UnknownHostException ||
                    e instanceof java.net.SocketTimeoutException)
                message = "Connection interrupted. Reconnect and tap Retry.";
            if (!stopped) UploadQueue.update(this, id, "failed", message);
        } finally {
            closeConnection();
            activeId = null;
            cancelled = null;
        }
    }

    private void transfer(JSONObject job, String id) throws Exception {
        String origin = job.getString("origin");
        if (!origin.equals(UploadQueue.origin(this))) throw new IOException("Reconnect to the original server and retry.");
        String cookie = CookieManager.getInstance().getCookie(origin + "/api/library/add");
        if (cookie == null) throw new IOException("Sign in to your server and retry.");
        JSONObject config = new JSONObject(get(origin + "/api/config", cookie));
        if (!job.getString("user").equals(config.optString("username")))
            throw new IOException("Sign in to the account that queued this upload.");
        String receipt = get(origin + "/api/library/imports/" + id, cookie);
        if ("null".equals(receipt.trim())) {
            post(job, origin, cookie, config.getString("token"));
        } else new JSONObject(receipt).getString("chapter");
    }

    private synchronized HttpURLConnection open(String address, String cookie) throws IOException {
        if (stopped || cancelled != null) throw new InterruptedIOException();
        connection = authenticatedConnection(address, cookie);
        return connection;
    }

    static HttpURLConnection authenticatedConnection(String address, String cookie) throws IOException {
        HttpURLConnection c = (HttpURLConnection)new URL(address).openConnection();
        c.setConnectTimeout(20000);
        c.setReadTimeout(6 * 60000);
        c.setInstanceFollowRedirects(false);
        c.setUseCaches(false);
        c.setRequestProperty("Cookie", cookie);
        return c;
    }

    private String get(String url, String cookie) throws IOException {
        try { return response(open(url, cookie)); }
        finally { closeConnection(); }
    }

    static String response(HttpURLConnection c) throws IOException {
        int status = c.getResponseCode();
        boolean ok = status >= 200 && status < 300;
        String text = "";
        try (InputStream in = ok ? c.getInputStream() : c.getErrorStream()) {
            if (in != null) {
                ByteArrayOutputStream buffer = new ByteArrayOutputStream();
                byte[] bytes = new byte[4096];
                int n;
                while (buffer.size() < 65536 && (n = in.read(bytes)) != -1) buffer.write(bytes, 0, n);
                text = buffer.toString("UTF-8");
            }
        }
        if (!ok) {
            String detail = "Server returned HTTP " + status + ". Retry the upload.";
            try { detail = new JSONObject(text).optString("error", detail); } catch (Exception ignored) {}
            if (status == 429 || status >= 500) throw new RetryableResponse(detail);
            throw new IOException(detail);
        }
        return text;
    }

    private byte[] bytes(String text) { return text.getBytes(StandardCharsets.UTF_8); }

    private void post(JSONObject job, String origin, String cookie, String token) throws Exception {
        String boundary = "Tralens" + UUID.randomUUID().toString().replace("-", "");
        byte[] prefix = bytes("--" + boundary + "\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n"
            + job.getJSONObject("metadata") + "\r\n");
        byte[] end = bytes("--" + boundary + "--\r\n");
        JSONArray files = job.getJSONArray("files");
        byte[][] headers = new byte[files.length()][];
        long length = prefix.length + end.length;
        for (int i = 0; i < files.length(); i++) {
            JSONObject file = files.getJSONObject(i);
            String name = file.getString("name").replaceAll("[\\p{Cntrl}\"\\\\/]", "_");
            headers[i] = bytes("--" + boundary + "\r\nContent-Disposition: form-data; name=\"page\"; filename=\""
                + name + "\"\r\nContent-Type: application/octet-stream\r\n\r\n");
            long size = file.getLong("size");
            if (size < 0) size = copy(file, null, 0, 0, "");
            if (size > LIMIT || length + size + headers[i].length + 2 > LIMIT)
                throw new IOException("Import up to 4 GiB at a time.");
            length += headers[i].length + size + 2;
        }
        HttpURLConnection c = open(origin + "/api/library/add", cookie);
        c.setRequestMethod("POST");
        c.setDoOutput(true);
        c.setRequestProperty("Origin", origin);
        c.setRequestProperty("X-Tralens-Token", token);
        c.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        c.setFixedLengthStreamingMode(length);
        long sent = prefix.length;
        String label = job.getJSONObject("metadata").optString("chapter", "Manga");
        try (OutputStream out = new BufferedOutputStream(c.getOutputStream(), 65536)) {
            out.write(prefix);
            for (int i = 0; i < files.length(); i++) {
                out.write(headers[i]);
                sent += headers[i].length;
                sent += copy(files.getJSONObject(i), out, sent, length, label);
                out.write('\r'); out.write('\n'); sent += 2;
            }
            out.write(end);
        }
        percent = 100;
        getSystemService(NotificationManager.class).notify(NOTICE, notification("Preparing pages on server", -1));
        new JSONObject(response(c)).getString("chapter");
    }

    private long copy(JSONObject file, OutputStream out, long sent, long total, String label) throws Exception {
        try (InputStream in = getContentResolver().openInputStream(Uri.parse(file.getString("uri")))) {
            if (in == null) throw new IOException("File unavailable. Check the document provider.");
            byte[] buf = new byte[65536];
            long read = 0;
            int n;
            while ((n = in.read(buf)) != -1) {
                if (stopped || cancelled != null || Thread.currentThread().isInterrupted()) throw new InterruptedIOException();
                read += n;
                if (read > LIMIT) throw new IOException("File exceeds 4 GiB.");
                if (out != null) {
                    out.write(buf, 0, n);
                    int progress = (int)Math.min(99, (sent + read) * 100 / total);
                    if (progress != percent) {
                        percent = progress;
                        getSystemService(NotificationManager.class).notify(NOTICE, notification(label, percent));
                    }
                }
            }
            return read;
        }
    }

    private synchronized void closeConnection() {
        HttpURLConnection c = connection;
        if (c != null) c.disconnect();
        connection = null;
    }

    @Override public void onTimeout(int startId, int type) { stopped = true; stopSelf(); }
    @Override public void onDestroy() {
        stopped = true;
        if (instance == this) instance = null;
        worker.shutdownNow();
        closeConnection();
        super.onDestroy();
    }
}
