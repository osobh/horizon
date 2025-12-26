#!/bin/bash
# Non-intrusive disk activity checker

echo "=== Disk Activity Diagnostic Report ==="
echo "Time: $(date)"
echo

echo "1. Currently running processes with disk/backup keywords:"
ps aux | grep -E "(snapshot|backup|rsync|borg|restic|timeshift|updatedb|apt|dpkg)" | grep -v grep | head -10

echo -e "\n2. Scheduled tasks:"
crontab -l 2>/dev/null || echo "No user crontab"

echo -e "\n3. System timers:"
systemctl list-timers --all | head -20

echo -e "\n4. VM-related processes:"
ps aux | grep -E "(qemu-ga|vmtoolsd|open-vm-tools|virtio)" | grep -v grep

echo -e "\n5. Current disk usage:"
df -h

echo -e "\n6. Memory and swap usage:"
free -h

echo -e "\n7. Load average:"
uptime

echo -e "\n8. Top 10 processes by CPU:"
ps aux --sort=-%cpu | head -11

echo -e "\n9. Recent disk-related log entries:"
sudo journalctl -n 50 | grep -i "disk\|backup\|snapshot" 2>/dev/null | tail -10

echo -e "\n10. Systemd maintenance timers:"
systemctl status apt-daily.timer apt-daily-upgrade.timer --no-pager 2>/dev/null | grep -E "(Active:|next trigger)"

echo -e "\n11. Check for I/O wait:"
top -b -n 1 | head -5

echo -e "\n12. Block device stats:"
cat /proc/diskstats | grep -E "sda|vda|xvda" | tail -5

echo -e "\n=== End of Report ==="