package app.tralens.reader;

import android.app.DownloadManager;
import android.content.SharedPreferences;
import com.getcapacitor.CapConfig;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.webkit.URLUtil;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebView;
import android.widget.Toast;
import com.getcapacitor.BridgeActivity;
import com.getcapacitor.BridgeWebViewClient;

public class MainActivity extends BridgeActivity {
    private String scheme, host;
    private int port;

    private boolean isReaderApi(Uri uri) {
        return scheme.equals(uri.getScheme()) && host.equals(uri.getHost())
            && (uri.getPort() == -1 ? defaultPort() : uri.getPort()) == port && uri.getPath() != null && uri.getPath().startsWith("/api/");
    }

    private int defaultPort() { return scheme.equals("https") ? 443 : 80; }

    @Override
    public void onCreate(Bundle state) {
        SharedPreferences prefs = getSharedPreferences("reader-server", MODE_PRIVATE);
        host = prefs.getString("host", "localhost");
        port = prefs.getInt("port", 7871);
        scheme = prefs.getString("scheme", "http");
        config = new CapConfig.Builder(this).setHostname(port == defaultPort() ? host : host + ":" + port)
            .setAndroidScheme(scheme).setLoggingEnabled(false).create();
        registerPlugin(ReaderConnection.class);
        registerPlugin(PhoneSignIn.class);
        super.onCreate(state);
        // Keep the entire reader viewport inside the system bars and camera cutout.
        View content = findViewById(android.R.id.content);
        ViewCompat.setOnApplyWindowInsetsListener(content, (view, windowInsets) -> {
            Insets safe = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars()
                | WindowInsetsCompat.Type.displayCutout());
            view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
            return windowInsets.inset(safe.left, safe.top, safe.right, safe.bottom);
        });
        ViewCompat.requestApplyInsets(content);
        // Capacitor serves the bundled reader. Let WebView send API requests,
        // including multipart bodies, to the selected same-origin backend.
        bridge.setWebViewClient(new BridgeWebViewClient(bridge) {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                if (isReaderApi(request.getUrl())) return null;
                return super.shouldInterceptRequest(view, request);
            }
        });
        if (!prefs.contains("host") && state == null) ReaderConnection.showSettings(this);
        bridge.getWebView().setDownloadListener((url, agent, disposition, mime, size) -> {
            Uri uri = Uri.parse(url);
            if (!isReaderApi(uri)) return;
            String filename = URLUtil.guessFileName(url, disposition, mime);
            DownloadManager.Request request = new DownloadManager.Request(uri)
                .setTitle(filename)
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
            String cookie = android.webkit.CookieManager.getInstance().getCookie(url);
            if (cookie != null && scheme.equals("https")) request.addRequestHeader("Cookie", cookie);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, filename);
            } else {
                request.setDestinationInExternalFilesDir(this, Environment.DIRECTORY_DOWNLOADS, filename);
            }
            try {
                getSystemService(DownloadManager.class).enqueue(request);
                Toast.makeText(this, "Downloading " + filename, Toast.LENGTH_SHORT).show();
            } catch (RuntimeException error) {
                Toast.makeText(this, "Could not start download.", Toast.LENGTH_LONG).show();
            }
        });
    }
}
