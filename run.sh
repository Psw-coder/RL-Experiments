#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_DIR"
# Arguments keep scripted runs noninteractive; no arguments opens the menu.
if [[ $# -eq 0 && -t 0 ]]; then
    while true; do
        printf '\n请选择 Breakout 实验：\n  1) RPO（默认）\n  2) PPO\n  q) 退出\n'
        if ! read -r -p '请输入选项 [1]: ' choice; then
            exit 0
        fi
        case "$choice" in
            1|rpo|RPO|'') set -- --algorithm rpo; break ;;
            2|ppo|PPO) set -- --algorithm ppo; break ;;
            q|Q) exit 0 ;;
            *) printf '无效选项，请输入 1、2 或 q。\n' ;;
        esac
    done
fi
exec "$VIRTUAL_ENV/bin/python" -B train.py "$@"
