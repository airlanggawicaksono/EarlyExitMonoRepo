import argparse
import subprocess
import os

def rewrite_nvpmodel_conf(cores, cpu_freq, gpu_freq):
    # Read the file up to the POWER_MODEL ID=0 definition
    with open('nvpmodel.conf', 'r') as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if line.startswith("< POWER_MODEL ID=0"):
            break
        out_lines.append(line)

    out_lines.append("< POWER_MODEL ID=0 NAME=DYNAMIC_MODE >\n")
    # Define CPU Cores
    for i in range(6):
        out_lines.append(f"CPU_ONLINE CORE_{i} {'1' if i < cores else '0'}\n")

    # Define CPU Freq ONLY for online cores
    # (Fixes EINVAL Error 22 when attempting to apply limits to offline cores)
    for i in range(cores):
        out_lines.append(f"CPU_A78_{i} MIN_FREQ 115200\n")
        out_lines.append(f"CPU_A78_{i} MAX_FREQ {int(cpu_freq) if int(cpu_freq)>0 else '-1'}\n")

    out_lines.append("GPU MIN_FREQ 0\n")
    out_lines.append(f"GPU MAX_FREQ {int(gpu_freq)*1000 if int(gpu_freq)>0 else '-1'}\n")
    out_lines.append("EMC MAX_FREQ 3199000000\n\n")
    out_lines.append("# mandatory section to configure the default power mode\n")
    out_lines.append("< PM_CONFIG DEFAULT=0 >\n")

    with open('nvpmodel.conf', 'w') as f:
        f.writelines(out_lines)

def set_config(cores, cpu_freq, gpu_freq):
    rewrite_nvpmodel_conf(cores, cpu_freq, gpu_freq)
    cmd = "sudo nvpmodel -f nvpmodel.conf -m 0 && sudo jetson_clocks && sudo systemctl restart jtop.service"
    subprocess.run(cmd, shell=True, text=True)

def reset_config():
    # Use the global system default settings
    cmd = "sudo nvpmodel -f /etc/nvpmodel.conf -m 0 && sudo jetson_clocks && sudo systemctl restart jtop.service"
    subprocess.run(cmd, shell=True, text=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, help='Number of active CPU cores')
    parser.add_argument('--cpu_freq', type=int, help='CPU frequency in kHz')
    parser.add_argument('--gpu_freq', type=int, help='GPU frequency in kHz')
    parser.add_argument('--reset', action='store_true', help='Reset device configuration to default')
    args = parser.parse_args()

    if args.reset:
        reset_config()
    else:
        if args.cores and args.cpu_freq and args.gpu_freq:
            set_config(args.cores, args.cpu_freq, args.gpu_freq)
        else:
            print("Please provide --cores, --cpu_freq, and --gpu_freq arguments, or use --reset.")
