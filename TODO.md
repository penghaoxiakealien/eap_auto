# TODO

- `auto_s_att.py` / `auto_NMH.py`: validation 选择逻辑需要复查。当前策略是“新 hypothesis 只要 validation 不比上一轮差就接受”；这会让并列最优或近似并列的更晚 hypothesis 覆盖更早、更稳的 hypothesis。`ioi_0312` 的 `Middle_Head/8.10_rerun_20260408_0615` 已出现这种现象：`initial/best` 的 test 明显优于最终 `final_hypothesis`。候选修正方向：加入 `eps` 容差，并在 validation 打平时优先保留更早 hypothesis。
