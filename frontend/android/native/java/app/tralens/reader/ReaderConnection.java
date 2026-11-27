package app.tralens.reader;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.security.NetworkSecurityPolicy;
import android.util.Patterns;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Locale;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

@CapacitorPlugin(name = "ReaderConnection")
public class ReaderConnection extends Plugin {
    @PluginMethod
    public void download(PluginCall call) {
        try {
            ((MainActivity) getActivity()).download(call.getString("path", ""), call.getString("ticket", ""),
                call.getString("filename", "page"));
            call.resolve();
        } catch (RuntimeException error) {
            call.reject("Could not start download.");
        }
    }

    @PluginMethod
    public void selected(PluginCall call) {
        SharedPreferences prefs = getContext().getSharedPreferences("reader-server", Context.MODE_PRIVATE);
        JSObject result = new JSObject();
        result.put("selected", prefs.contains("host"));
        call.resolve(result);
    }

    @PluginMethod
    public void connect(PluginCall call) {
        getActivity().runOnUiThread(() -> {
            try {
                saveServer(getActivity(), call.getString("url", ""));
                call.resolve();
                getActivity().recreate();
            } catch (IllegalArgumentException error) {
                call.reject(error.getMessage());
            }
        });
    }

    private static void saveServer(Activity activity, String value) {
        URI url;
        try {
            url = new URI(value.trim()).parseServerAuthority();
        } catch (URISyntaxException invalid) {
            throw new IllegalArgumentException("Enter a full HTTP or HTTPS server URL.");
        }
        String scheme = String.valueOf(url.getScheme()).toLowerCase(Locale.ROOT);
        String host = url.getHost();
        boolean tls = scheme.equals("https");
        if (host == null || (!tls && !scheme.equals("http")) || url.getUserInfo() != null
                || url.getQuery() != null || url.getFragment() != null
                || !(url.getPath().isEmpty() || url.getRawPath().equals("/"))) {
            throw new IllegalArgumentException("Enter the server URL without a page path, such as https://manga.example.com.");
        }
        host = host.toLowerCase(Locale.ROOT);
        boolean dns = tls && !host.matches("[0-9.]+") && Patterns.DOMAIN_NAME.matcher(host).matches();
        if (!dns && !host.equals("localhost") && (!host.matches("[0-9.]+") || !Patterns.IP_ADDRESS.matcher(host).matches()
                || host.startsWith("0.") || host.matches(".*(?:^|\\.)0[0-9]+(?:\\.|$).*"))) {
            throw new IllegalArgumentException("Use an IPv4 address, or HTTPS for a hostname.");
        }
        int port = url.getPort() == -1 ? (tls ? 443 : 80) : url.getPort();
        if (port < 1 || port > 65535 || url.getRawAuthority().endsWith(":")) {
            throw new IllegalArgumentException("Use a port from 1 to 65535.");
        }
        if (!tls && (Build.VERSION.SDK_INT < 24
                || !NetworkSecurityPolicy.getInstance().isCleartextTrafficPermitted(host))) {
            throw new IllegalArgumentException("Use HTTPS or an APK built with this server’s HTTP address.");
        }
        if (!tls && !host.equals("localhost") && !host.equals("127.0.0.1")) {
            throw new IllegalArgumentException("Use HTTPS to sign in to a remote server.");
        }
        activity.getSharedPreferences("reader-server", Context.MODE_PRIVATE).edit()
            .putString("host", host).putInt("port", port).putString("scheme", scheme).apply();
    }
}
