package app.tralens.reader;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import androidx.activity.result.ActivityResult;
import androidx.core.content.ContextCompat;
import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import org.json.JSONArray;
import org.json.JSONObject;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@CapacitorPlugin(name = "ReaderUploads")
public class ReaderUploads extends Plugin {
    // Only documents granted by the Android picker can be queued by the web UI.
    private final Map<String, JSONObject> selection = new HashMap<>();

    @PluginMethod public void choose(PluginCall call) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
            .setType("*/*").putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(call, intent, "picked");
    }

    @ActivityCallback private synchronized void picked(PluginCall call, ActivityResult result) {
        if (call == null) return;
        JSArray selected = new JSArray();
        try {
            Intent data = result.getData();
            if (result.getResultCode() == Activity.RESULT_OK && data != null) {
                int count = data.getClipData() == null ? 1 : data.getClipData().getItemCount();
                if (count > 256) throw new IllegalArgumentException("Choose up to 256 files per import.");
                for (int i = 0; i < count; i++) {
                    Uri uri = data.getClipData() == null ? data.getData() : data.getClipData().getItemAt(i).getUri();
                    if (uri == null || !"content".equals(uri.getScheme()) ||
                        (getContext().getPackageName() + ".fileprovider").equals(uri.getAuthority()))
                        throw new IllegalArgumentException("Choose a document from Files.");
                    try (Cursor cursor = getContext().getContentResolver().query(uri,
                            new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE}, null, null, null)) {
                        if (cursor == null || !cursor.moveToFirst()) throw new IllegalArgumentException("Cannot read the selected file.");
                        String key = UUID.randomUUID().toString();
                        String name = cursor.getString(0);
                        if (name == null || name.isEmpty()) throw new IllegalArgumentException();
                        long size = cursor.isNull(1) ? -1 : cursor.getLong(1);
                        selected.put(new JSONObject().put("uri", uri.toString()).put("handle", key).put("name", name).put("size", size));
                        selection.put(key, new JSONObject().put("uri", uri.toString()).put("name", name).put("size", size));
                    }
                }
            }
            call.resolve(new JSObject().put("files", selected));
        } catch (Exception e) { call.reject("Could not read the selection. Choose up to 256 files from Files."); }
    }

    @PluginMethod public synchronized void clearSelection(PluginCall call) {
        selection.clear();
        call.resolve();
    }

    @PluginMethod public synchronized void enqueue(PluginCall call) {
        String id = null;
        try {
            String origin = UploadQueue.origin(getContext());
            if (!origin.equals(call.getString("origin"))) throw new IllegalArgumentException();
            JSObject meta = call.getObject("metadata");
            id = meta.getString("import_id");
            JSONArray handles = call.getArray("handles"), files = new JSONArray();
            if (!id.matches("[0-9a-f]{32}") || handles.length() < 1 || handles.length() > 256)
                throw new IllegalArgumentException();
            for (int i = 0; i < handles.length(); i++) {
                JSONObject file = selection.get(handles.getString(i));
                if (file == null) throw new IllegalArgumentException();
                files.put(file);
            }
            UploadQueue.add(getContext(), new JSONObject().put("id", id).put("origin", origin)
                .put("user", call.getString("user")).put("metadata", meta).put("files", files).put("state", "queued"));
        } catch (Exception e) { call.reject("Could not queue these files. Choose them again in Files."); return; }
        try {
            start(getContext());
            if (android.os.Build.VERSION.SDK_INT >= 33 && !getContext().getSharedPreferences("reader-uploads", 0)
                    .getBoolean("notification-requested", false)) {
                getContext().getSharedPreferences("reader-uploads", 0).edit().putBoolean("notification-requested", true).apply();
                getActivity().runOnUiThread(() -> getActivity().requestPermissions(
                    new String[]{android.Manifest.permission.POST_NOTIFICATIONS}, 701));
            }
        } catch (RuntimeException e) {
            try { UploadQueue.update(getContext(), id, "failed", "Android paused this upload. Tap Retry."); }
            catch (Exception ignored) {}
        }
        call.resolve();
    }

    static void start(Context context) {
        ContextCompat.startForegroundService(context, new Intent(context, UploadService.class));
    }

    @PluginMethod public void list(PluginCall call) {
        try {
            JSONArray jobs = UploadQueue.jobs(getContext()), result = new JSONArray();
            for (int i = 0; i < jobs.length(); i++) {
                JSONObject job = jobs.getJSONObject(i);
                if (!belongs(job, call)) continue;
                JSONObject meta = job.getJSONObject("metadata");
                JSONObject item = new JSONObject().put("id", job.getString("id")).put("title", meta.optString("title"))
                    .put("chapter", meta.optString("chapter")).put("book", meta.optString("book"))
                    .put("state", job.getString("state")).put("message", job.optString("message"));
                if ("queued".equals(job.optString("state")) && !UploadService.running()) item.put("state", "paused");
                if (job.getString("id").equals(UploadService.activeId))
                    item.put("state", "uploading").put("percent", UploadService.percent);
                result.put(item);
            }
            call.resolve(new JSObject().put("jobs", result));
        } catch (Exception e) { call.reject("Could not read the upload queue."); }
    }

    private boolean belongs(JSONObject job, PluginCall call) {
        return UploadQueue.origin(getContext()).equals(job.optString("origin"))
            && job.optString("user").equals(call.getString("user"));
    }

    @PluginMethod public void change(PluginCall call) {
        try {
            JSONArray jobs = UploadQueue.jobs(getContext());
            for (int i = 0; i < jobs.length(); i++) {
                JSONObject job = jobs.getJSONObject(i);
                if (!belongs(job, call) || !job.getString("id").equals(call.getString("id"))) continue;
                String id = job.getString("id");
                if (!Boolean.TRUE.equals(call.getBoolean("retry"))) {
                    UploadQueue.update(getContext(), id, null, "");
                    UploadService.cancel(id);
                } else {
                    if ("failed".equals(job.optString("state"))) UploadQueue.update(getContext(), id, "queued", "");
                    start(getContext());
                }
            }
            call.resolve();
        } catch (Exception e) { call.reject("Could not update the upload. Reopen the queue and retry."); }
    }
}
