package app.tralens.reader;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import androidx.core.content.ContextCompat;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.File;
import java.io.IOException;
import java.util.UUID;

@CapacitorPlugin(name = "ReaderDownloads")
public class ReaderDownloads extends Plugin {
    static final Object LOCK = new Object();
    static SharedPreferences store(Context c) { return c.getSharedPreferences("offline-transfers", Context.MODE_PRIVATE); }

    static JSONObject lookup(Context c, String key) throws JSONException {
        String raw = store(c).getString(key, null);
        return raw == null ? null : new JSONObject(raw);
    }

    static void write(Context c, JSONObject job) throws Exception {
        if (!store(c).edit().putString(job.getString("generation"), job.toString()).commit())
            throw new IOException("Could not save download progress. Check device storage.");
    }

    static File directory(Context c, String generation) {
        // The server, username and directory are bound together when the job is created.
        UUID id = UUID.fromString(generation);
        if (!id.toString().equals(generation)) throw new IllegalArgumentException();
        return new File(new File(c.getNoBackupFilesDir(), "offline-pages"), generation);
    }

    private boolean belongs(JSONObject job, PluginCall call) {
        return job != null && UploadQueue.origin(getContext()).equals(job.optString("origin"))
            && job.optString("user").equals(call.getString("user"));
    }

    @PluginMethod public void start(PluginCall call) {
        try {
            Context c = getContext();
            String key = call.getString("generation", ""), origin = UploadQueue.origin(c);
            if (!origin.equals(call.getString("origin"))) throw new IllegalArgumentException();
            synchronized (LOCK) {
                JSONObject job = lookup(c, key);
                if (job == null) {
                    JSONArray urls = call.getArray("urls");
                    int offset = call.getInt("offset", 0);
                    if (urls == null || urls.length() > 256 || offset < 0 || offset >= urls.length())
                        throw new IllegalArgumentException();
                    for (int i = 0; i < urls.length(); i++)
                        if (!urls.getString(i).matches("/api/library/pages/[0-9a-f]{32}/(?:original|translated)(?:\\?v=[0-9]+)?"))
                            throw new IllegalArgumentException();
                    File dir = directory(c, key);
                    if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException();
                    job = new JSONObject().put("generation", key).put("origin", origin).put("user", call.getString("user"))
                        .put("title", call.getString("title")).put("urls", urls).put("saved", offset)
                        .put("created", System.currentTimeMillis());
                } else if (!belongs(job, call)) throw new IllegalArgumentException();
                if (job.getInt("saved") == job.getJSONArray("urls").length()) { call.resolve(); return; }
                if (!"queued".equals(job.optString("state"))) job.put("attempt", UUID.randomUUID().toString());
                job.put("state", "queued").remove("error");
                write(c, job);
            }
            ContextCompat.startForegroundService(c, new Intent(c, OfflineDownloadService.class));
            ReaderUploads.requestTransferNotifications(getActivity());
            call.resolve();
        } catch (Exception e) { call.reject("Could not start the download. Check device storage and tap Resume."); }
    }

    @PluginMethod public void status(PluginCall call) {
        try {
            JSObject jobs = new JSObject();
            synchronized (LOCK) {
                for (String key : store(getContext()).getAll().keySet()) {
                    JSONObject job = lookup(getContext(), key);
                    if (!belongs(job, call)) continue;
                    String state = job.getString("state");
                    if ("queued".equals(state) && !OfflineDownloadService.running) state = "paused";
                    jobs.put(key, new JSObject().put("saved", job.getInt("saved")).put("state", state)
                        .put("error", job.optString("error")));
                }
            }
            call.resolve(new JSObject().put("jobs", jobs));
        } catch (Exception e) { call.reject("Could not read saved downloads."); }
    }

    @PluginMethod public void change(PluginCall call) {
        try {
            Context c = getContext();
            String key = call.getString("generation", "");
            synchronized (LOCK) {
                JSONObject job = lookup(c, key);
                if (job == null) { call.resolve(); return; }
                if (!belongs(job, call)) throw new IllegalArgumentException();
                OfflineDownloadService.interrupt(job.optString("attempt"));
                if (Boolean.TRUE.equals(call.getBoolean("remove"))) {
                    File dir = directory(c, key);
                    File[] files = dir.listFiles();
                    if (files != null) for (File f : files) if (f.exists() && !f.delete()) throw new IOException();
                    if (dir.exists() && !dir.delete()) throw new IOException();
                    if (!store(c).edit().remove(key).commit()) throw new IOException();
                } else if (!"complete".equals(job.optString("state"))) {
                    job.put("state", "paused").remove("error");
                    write(c, job);
                }
            }
            call.resolve();
        } catch (Exception e) { call.reject("Could not change the download."); }
    }

    @PluginMethod public void page(PluginCall call) {
        try {
            Context c = getContext();
            String key = call.getString("generation", "");
            int index = call.getInt("index", -1);
            synchronized (LOCK) {
                JSONObject job = lookup(c, key);
                if (!belongs(job, call) || index < 0 || index >= job.getInt("saved")) throw new IOException();
                for (String ext : new String[]{".png", ".jpg", ".webp"}) {
                    File page = new File(directory(c, key), index + ext);
                    if (page.isFile()) { call.resolve(new JSObject().put("path", page.getAbsolutePath())); return; }
                }
            }
        } catch (Exception ignored) {}
        call.reject("Page unavailable on this device.");
    }
}
