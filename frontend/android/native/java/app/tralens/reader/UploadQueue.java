package app.tralens.reader;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/** One small, durable manifest per import. Page bytes stay with the document provider. */
final class UploadQueue {
    private static SharedPreferences store(Context c) {
        return c.getSharedPreferences("reader-uploads", Context.MODE_PRIVATE);
    }

    static synchronized JSONArray jobs(Context c) throws JSONException {
        return new JSONArray(store(c).getString("jobs", "[]"));
    }

    private static void save(Context c, JSONArray jobs) throws IOException {
        if (!store(c).edit().putString("jobs", jobs.toString()).commit())
            throw new IOException("Could not save the upload queue.");
    }

    static synchronized void add(Context c, JSONObject job) throws Exception {
        JSONArray jobs = jobs(c);
        JSONArray files = job.getJSONArray("files");
        Set<Uri> acquired = new HashSet<>();
        try {
            for (int i = 0; i < files.length(); i++) {
                Uri uri = Uri.parse(files.getJSONObject(i).getString("uri"));
                c.getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                acquired.add(uri);
            }
            save(c, jobs.put(job));
        } catch (Exception error) {
            releaseUnused(c, acquired, jobs(c));
            throw error;
        }
    }

    static synchronized void update(Context c, String id, String state, String message) throws Exception {
        JSONArray jobs = jobs(c);
        for (int i = 0; i < jobs.length(); i++) {
            JSONObject job = jobs.getJSONObject(i);
            if (!id.equals(job.getString("id"))) continue;
            if (state != null) {
                job.put("state", state).put("message", message);
                save(c, jobs);
            } else {
                Set<Uri> removed = new HashSet<>();
                collectUris(job, removed);
                jobs.remove(i);
                save(c, jobs);
                releaseUnused(c, removed, jobs);
            }
            return;
        }
    }

    private static void collectUris(JSONObject job, Set<Uri> set) throws JSONException {
        JSONArray files = job.getJSONArray("files");
        for (int i = 0; i < files.length(); i++)
            set.add(Uri.parse(files.getJSONObject(i).getString("uri")));
    }

    private static void releaseUnused(Context c, Set<Uri> removed, JSONArray remaining) throws JSONException {
        Set<Uri> used = new HashSet<>();
        for (int i = 0; i < remaining.length(); i++) collectUris(remaining.getJSONObject(i), used);
        removed.removeAll(used);
        for (Uri uri : removed) {
            try { c.getContentResolver().releasePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION); }
            catch (SecurityException ignored) { /* Provider already revoked the grant. */ }
        }
    }

    static String origin(Context c) {
        SharedPreferences prefs = c.getSharedPreferences("reader-server", Context.MODE_PRIVATE);
        String scheme = prefs.getString("scheme", "http");
        int port = prefs.getInt("port", 7871);
        return scheme + "://" + prefs.getString("host", "localhost")
            + (port == ("https".equals(scheme) ? 443 : 80) ? "" : ":" + port);
    }
}
