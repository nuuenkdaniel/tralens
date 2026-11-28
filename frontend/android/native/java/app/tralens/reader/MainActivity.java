package app.tralens.reader;

import android.app.DownloadManager;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import com.getcapacitor.CapConfig;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import androidx.activity.OnBackPressedCallback;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowCompat;
import android.webkit.CookieManager;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebView;
import com.getcapacitor.BridgeActivity;
import com.getcapacitor.BridgeWebViewClient;

public class MainActivity extends BridgeActivity {
    private String scheme, host;
    private int port;

    private boolean isReaderOrigin(Uri uri) {
        return scheme.equals(uri.getScheme()) && host.equals(uri.getHost())
            && (uri.getPort() == -1 ? defaultPort() : uri.getPort()) == port;
    }

    private boolean isReaderApi(Uri uri) {
        return isReaderOrigin(uri) && uri.getPath() != null && uri.getPath().startsWith("/api/");
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
        registerPlugin(ReaderUploads.class);
        registerPlugin(PhoneSignIn.class);
        super.onCreate(state);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                bridge.getWebView().evaluateJavascript(
                    "dispatchEvent(new Event('tralens-back', {cancelable:true}))", result -> {
                        if ("false".equals(result)) return;
                        setEnabled(false);
                        getOnBackPressedDispatcher().onBackPressed();
                        setEnabled(true);
                    });
            }
        });
        // Keep the entire reader viewport inside the system bars and camera cutout.
        View content = findViewById(android.R.id.content);
        ViewCompat.setOnApplyWindowInsetsListener(content, (view, windowInsets) -> {
            Insets safe = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars()
                | WindowInsetsCompat.Type.displayCutout());
            view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
            return windowInsets.inset(safe.left, safe.top, safe.right, safe.bottom);
        });
        boolean lightSystem = (getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK)
            != Configuration.UI_MODE_NIGHT_YES;
        WindowCompat.getInsetsController(getWindow(), content).setAppearanceLightStatusBars(lightSystem);
        ViewCompat.requestApplyInsets(content);
        // Capacitor serves the bundled reader. Let WebView send API requests,
        // including multipart bodies, to the selected same-origin backend.
        bridge.setWebViewClient(new BridgeWebViewClient(bridge) {
            @Override
            public void onPageFinished(WebView view, String url) {
                super.onPageFinished(view, url);
                // Login and logout both reload the reader.
                CookieManager.getInstance().flush();
                openQueueIfRequested(view);
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                if (request.isForMainFrame() && isReaderOrigin(uri) && "/privacy".equals(uri.getPath())) {
                    try {
                        startActivity(new Intent(Intent.ACTION_VIEW, uri));
                    } catch (ActivityNotFoundException ignored) { }
                    return true;
                }
                return super.shouldOverrideUrlLoading(view, request);
            }

            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                if (request.isForMainFrame() && !(isReaderOrigin(uri) &&
                        ("/".equals(uri.getPath()) || "/index.html".equals(uri.getPath())))) {
                    return new WebResourceResponse("text/plain", "UTF-8", null);
                }
                if (!request.isForMainFrame() && isReaderApi(uri)) return null;
                return super.shouldInterceptRequest(view, request);
            }
        });
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (bridge != null && bridge.getWebView().getProgress() == 100) openQueueIfRequested(bridge.getWebView());
    }

    private void openQueueIfRequested(WebView view) {
        if ("app.tralens.reader.QUEUE".equals(getIntent().getAction())) {
            getIntent().setAction(null);
            view.evaluateJavascript("location.hash = 'queue'", null);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        CookieManager.getInstance().flush();
    }

    void download(String path, String ticket, String name) {
        if (!path.matches("/api/library/(books/[0-9a-f]{32}/(both|original|translated)\\.zip|pages/[0-9a-f]{32}/(original|translated))"))
            throw new IllegalArgumentException("Choose a reader download.");
        Uri uri = new Uri.Builder().scheme(scheme).encodedAuthority(host + ":" + port).path(path).build();
        String filename = name.replaceAll("[^\\p{L}\\p{N} ._-]", "_").replaceFirst("^[. ]+", "");
        if (filename.isEmpty()) filename = "page";
        // Keep names within Android's byte-based filesystem limit, retaining the extension.
        if (filename.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > 240) {
            int dot = filename.lastIndexOf('.');
            String ext = dot >= 0 && filename.length() - dot <= 8 ? filename.substring(dot) : "";
            filename = filename.substring(0, filename.offsetByCodePoints(0, Math.min(56, filename.codePointCount(0, filename.length())))) + ext;
        }
        DownloadManager.Request request = new DownloadManager.Request(uri)
            .setTitle(filename)
            .addRequestHeader("X-Tralens-Download", ticket)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
            request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, filename);
        else
            request.setDestinationInExternalFilesDir(this, Environment.DIRECTORY_DOWNLOADS, filename);
        getSystemService(DownloadManager.class).enqueue(request);
    }
}
