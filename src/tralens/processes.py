"""Linux process ownership for bounded private workers; never PID-only signaling."""
import os
from pathlib import Path
import signal


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':int(fields[19]),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError,ProcessLookupError):return None


def group_members(group):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=group or fields[0]=='Z':continue
            members.append((int(path.name),(path/'environ').read_bytes().split(b'\0')))
        except (FileNotFoundError,ProcessLookupError):continue
    return members


def recover(state):
    """Recover only a dead supervisor's token-marked group, including descendants."""
    owner=state.get('owner');token=state.get('token');group=state.get('process_group')
    if not owner or not token or type(group) is not int or group<=1:
        return 'legacy_unverifiable'
    if owner.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return 'previous_boot'
    members=group_members(group)
    if not members:return 'already_stopped'
    if identity(owner['pid'])==owner:return 'live_supervisor'
    marker=('TRALENS_WORKER_TOKEN='+token).encode()
    if any(marker not in environ for _,environ in members):return 'identity_mismatch'
    try:os.killpg(group,signal.SIGKILL)
    except ProcessLookupError:pass
    return 'owned_group_killed'
