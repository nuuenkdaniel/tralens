"""Generate Android's host-specific HTTP permissions inside the disposable project."""
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

res = Path(sys.argv[1])
hosts = {"localhost", "127.0.0.1"}
configured = os.environ.get("TRALENS_HTTP_HOSTS")
if configured is None:
    interfaces = json.loads(subprocess.check_output(["ip", "-j", "-4", "addr", "show", "up"]))
    addresses = [item["local"] for interface in interfaces
                 if not interface["ifname"].startswith(("docker", "br-", "veth"))
                 for item in interface["addr_info"]]
else:
    addresses = configured.split()
for address in addresses:
    ip = ipaddress.IPv4Address(address)
    if not ip.is_unspecified and not ip.is_multicast:
        hosts.add(str(ip))
policy = ET.Element("network-security-config")
ET.SubElement(policy, "base-config", cleartextTrafficPermitted="false")
allowed = ET.SubElement(policy, "domain-config", cleartextTrafficPermitted="true")
for host in sorted(hosts):
    ET.SubElement(allowed, "domain").text = host
ET.ElementTree(policy).write(res/"xml/network_security.xml", encoding="utf-8", xml_declaration=True)
