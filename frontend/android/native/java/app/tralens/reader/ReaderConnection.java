package app.tralens.reader;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.security.NetworkSecurityPolicy;
import android.text.InputType;
import android.util.Patterns;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.app.AlertDialog;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

@CapacitorPlugin(name = "ReaderConnection")
public class ReaderConnection extends Plugin {
    @PluginMethod
    public void configure(PluginCall call) {
        getActivity().runOnUiThread(() -> {
            showSettings(getActivity());
            call.resolve();
        });
    }

    private static EditText field(LinearLayout layout, String label, String value, int inputType) {
        Context context = layout.getContext();
        TextView caption = new TextView(context);
        caption.setText(label);
        EditText input = new EditText(context);
        input.setId(android.view.View.generateViewId());
        caption.setLabelFor(input.getId());
        input.setSingleLine(true);
        input.setInputType(inputType);
        input.setText(value);
        input.setSelectAllOnFocus(true);
        layout.addView(caption);
        layout.addView(input);
        return input;
    }

    static void showSettings(Activity activity) {
        SharedPreferences prefs = activity.getSharedPreferences("reader-server", Context.MODE_PRIVATE);
        LinearLayout layout = new LinearLayout(activity);
        layout.setOrientation(LinearLayout.VERTICAL);
        int pad = Math.round(24 * activity.getResources().getDisplayMetrics().density);
        layout.setPadding(pad, pad / 2, pad, 0);
        EditText ip = field(layout, "Server address", prefs.getString("host", ""),
            InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        ip.setHint("192.168.1.100");
        EditText port = field(layout, "Port", String.valueOf(prefs.getInt("port", 7871)), InputType.TYPE_CLASS_NUMBER);
        CheckBox tls = new CheckBox(activity);
        tls.setText("Use HTTPS");
        tls.setChecked("https".equals(prefs.getString("scheme", "http")));
        layout.addView(tls);
        AlertDialog dialog = new AlertDialog.Builder(activity).setTitle("Connect to Tralens")
            .setMessage("Enter your server’s IPv4 address or HTTPS hostname.")
            .setView(layout).setNegativeButton("Cancel", null).setPositiveButton("Connect", null).create();
        dialog.show();
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
            String host = ip.getText().toString().trim();
            boolean dns = tls.isChecked() && !host.matches("[0-9.]+") && Patterns.DOMAIN_NAME.matcher(host).matches();
            if (!dns && !host.equals("localhost") && (!host.matches("[0-9.]+") || !Patterns.IP_ADDRESS.matcher(host).matches()
                    || host.startsWith("0.") || host.matches(".*(?:^|\\.)0[0-9]+(?:\\.|$).*"))) {
                ip.setError("Enter an IPv4 address, or enable HTTPS for a hostname.");
                return;
            }
            int number;
            try { number = Integer.parseInt(port.getText().toString()); }
            catch (NumberFormatException invalid) { number = 0; }
            if (number < 1 || number > 65535) {
                port.setError("Enter a port from 1 to 65535.");
                return;
            }
            if (!tls.isChecked() && (Build.VERSION.SDK_INT < 24
                    || !NetworkSecurityPolicy.getInstance().isCleartextTrafficPermitted(host))) {
                ip.setError("This IP requires HTTPS or an APK built with this server’s HTTP address.");
                return;
            }
            prefs.edit().putString("host", host).putInt("port", number)
                .putString("scheme", tls.isChecked() ? "https" : "http").apply();
            dialog.dismiss();
            activity.recreate();
        });
    }
}
