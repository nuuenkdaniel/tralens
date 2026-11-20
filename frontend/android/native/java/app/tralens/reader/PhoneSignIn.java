package app.tralens.reader;

import android.content.Intent;
import android.net.Uri;
import android.os.SystemClock;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

@CapacitorPlugin(name = "PhoneSignIn")
public class PhoneSignIn extends Plugin {
    private ServerSocket listener;

    @PluginMethod
    public synchronized void prepare(PluginCall call) {
        closeListener();
        try {
            listener = new ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"));
            JSObject result = new JSObject();
            result.put("port", listener.getLocalPort());
            call.resolve(result);
        } catch (IOException error) {
            call.reject("Could not prepare phone sign-in.");
        }
    }

    @PluginMethod
    public synchronized void cancel(PluginCall call) {
        if (listener != null && Integer.valueOf(listener.getLocalPort()).equals(call.getInt("port")))
            closeListener();
        call.resolve();
    }

    private synchronized void closeListener() {
        if (listener != null) {
            try { listener.close(); } catch (IOException ignored) {}
            listener = null;
        }
    }

    @Override protected void handleOnDestroy() {
        closeListener();
        super.handleOnDestroy();
    }

    @PluginMethod
    public synchronized void authorize(PluginCall call) {
        final ServerSocket socket = listener;
        Uri uri = Uri.parse(call.getString("url", ""));
        String redirect = socket == null ? "" : "http://127.0.0.1:" + socket.getLocalPort() + "/auth/callback";
        if (socket == null || !"https".equals(uri.getScheme()) || !"auth.openai.com".equals(uri.getEncodedAuthority())
                || !"/api/accounts/authorize".equals(uri.getEncodedPath())
                || !redirect.equals(uri.getQueryParameter("redirect_uri")) || uri.getQueryParameter("state") == null) {
            call.reject("Invalid ChatGPT sign-in address.");
            return;
        }
        try {
            getActivity().startActivity(new Intent(Intent.ACTION_VIEW, uri).addCategory(Intent.CATEGORY_BROWSABLE));
        } catch (RuntimeException error) {
            closeListener();
            call.reject("Open a system browser to sign in.");
            return;
        }
        new Thread(() -> {
            try {
                call.resolve(receive(socket, uri.getQueryParameter("state")));
            } catch (IOException error) {
                call.reject("Phone sign-in ended. Select the model again to retry.");
            } finally {
                synchronized (this) {
                    if (listener == socket) closeListener();
                }
            }
        }, "tralens-phone-sign-in").start();
    }

    private JSObject receive(ServerSocket server, String state) throws IOException {
        long end = SystemClock.elapsedRealtime() + TimeUnit.MINUTES.toMillis(5);
        String host = "127.0.0.1:" + server.getLocalPort();
        while (SystemClock.elapsedRealtime() < end) {
            server.setSoTimeout((int)Math.max(1, end - SystemClock.elapsedRealtime()));
            try (Socket client = server.accept()) {
                client.setSoTimeout(3000);
                String headers = readHeaders(client);
                String[] lines = headers.split("\r\n");
                String[] request = lines[0].split(" ");
                boolean correctHost = false;
                for (String line : lines) if (line.equalsIgnoreCase("Host: " + host)) correctHost = true;
                Uri target = request.length == 3 ? Uri.parse(request[1]) : Uri.EMPTY;
                if (!correctHost || !request[0].equals("GET") || !target.isRelative() || target.getAuthority() != null
                        || !"/auth/callback".equals(target.getEncodedPath())
                        || !state.equals(target.getQueryParameter("state"))) {
                    client.getOutputStream().write("HTTP/1.0 400 Bad Request\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
                    continue;
                }
                JSObject result = new JSObject();
                for (String key : new String[]{"state", "code", "client_id", "error"}) {
                    if (target.getQueryParameters(key).size() > 1) throw new IOException("Duplicate field");
                    String value = target.getQueryParameter(key);
                    if (value != null) result.put(key, value);
                }
                String body = "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                    + "<title>Return to Tralens</title><style>body{color-scheme:dark;background:#121a18;color:#edf7f2;font:18px/1.5 system-ui;"
                    + "margin:0;min-height:100svh;display:grid;place-items:center}main{padding:28px;max-width:26rem}a{color:#98e0c7}</style>"
                    + "<script>history.replaceState(null,'','/auth/result')</script><main><h1>Return to Tralens</h1>"
                    + "<p>Switch back to the reader to see your sign-in result.</p><a href='tralens-reader://signin/return'>Open Tralens</a></main>";
                byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
                String response = "HTTP/1.0 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\n"
                    + "Referrer-Policy: no-referrer\r\nContent-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-ancestors 'none'\r\n"
                    + "Content-Length: " + bytes.length + "\r\n\r\n";
                client.getOutputStream().write(response.getBytes(StandardCharsets.US_ASCII));
                client.getOutputStream().write(bytes);
                return result;
            } catch (SocketTimeoutException ignored) {
                // A browser may preconnect without sending a request.
            }
        }
        throw new IOException("Expired");
    }

    private String readHeaders(Socket socket) throws IOException {
        BufferedInputStream input = new BufferedInputStream(socket.getInputStream());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int tail = 0;
        while (out.size() < 16384) {
            int value = input.read();
            if (value < 0) break;
            out.write(value);
            tail = (tail << 8) | value;
            if (tail == 0x0d0a0d0a) return out.toString("US-ASCII");
        }
        return "";
    }
}
