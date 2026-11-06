# Quy trình commit

Rút ra từ lịch sử git của repo (nhánh `feat/`, `docs/`, `fix/`, `chore/`, commit merge dạng `Merge <nhánh>`). Phần "Cách làm" là gợi ý, chỉnh lại nếu bạn muốn khác.

## Nguyên tắc

1. **Mỗi commit một mục đích.** Đừng gộp tính năng code, tài liệu và notebook vào cùng một commit.
2. **Tách trước khi commit, không tách sau.** Xong một bước nhỏ thì commit ngay, đừng để dồn nhiều thay đổi trong working tree.
3. **Code và tài liệu của nó có thể nằm chung nhánh nhưng khác commit**, để dễ revert riêng.
4. **Notebook (`.ipynb`) commit riêng**, vì output và `execution_count` làm diff rất nhiễu. Không trộn với code thư viện.
5. **Không commit file sinh ra**: thư mục `runs/`, `.pt`, `.npy` (đã có trong `.gitignore`).

## Đặt tên nhánh

| Loại | Dạng | Ví dụ có sẵn |
|---|---|---|
| Tính năng | `feat/<tên>` | `feat/notebooks`, `feat/pacman-rl-library` |
| Tài liệu | `docs/<tên>` | `docs/html-documentation`, `docs/report` |
| Sửa lỗi | `fix/<tên>` | `fix/ppo-save-dir` |
| Việc vặt | `chore/<tên>` | `chore/gitignore` |

Không commit thẳng lên `main`: tạo nhánh, commit, rồi merge vào `main` bằng commit `Merge <nhánh>` (merge commit, không squash).

## Thông điệp commit

- Tiếng Anh, động từ nguyên mẫu, viết hoa chữ đầu, không chấm cuối: `Add ...`, `Create ...`, `Update: ...`, `Fix ...`.
- Dòng đầu nói **cái gì thay đổi**, không nói file nào.
- Có `Co-Authored-By` ở cuối nếu Claude tham gia viết.

## Cách làm

```bash
git status --short            # xem toàn bộ thay đổi trước
git switch -c feat/<tên>      # tạo nhánh nếu chưa có

git add <file hoặc thư mục>   # chỉ add đúng nhóm của commit này
git diff --cached             # đọc lại những gì sắp commit
git commit -m "..."
```

Một file có hai loại thay đổi trộn nhau thì dùng `git add -p` để chọn từng đoạn.

Xong nhánh:

```bash
git switch main
git merge --no-ff feat/<tên>   # ra commit "Merge feat/<tên>"
```

## Ví dụ: tách nhóm thay đổi hiện tại

Các thay đổi chưa commit (reward `death_penalty`, theo dõi số lần chết, tài liệu) nên tách thành các commit sau, theo thứ tự:

1. **`feat/death-penalty`**: `Add death_penalty and life_lost signal to PPO and DQN`
   - `envs/wrappers.py` (`info["life_lost"]`)
   - `envs/vector.py` (`pop_deaths`, `recv` trả thêm `died`)
   - `agents/ppo.py`, `agents/dqn.py` (trường `death_penalty`, nhận `died` từ `recv`)
2. **Cùng nhánh**: `Track deaths and per-life stats in training log`
   - `agents/ppo.py`, `agents/dqn.py` (`log["deaths"]`, dòng in theo lượt chơi)
   - `agents/common.py`, `runs.py` (docstring, nối run có `deaths`)
3. **`docs/death-penalty`**: `Document death_penalty, died flag and deaths log`
   - `pacman_rl/docs/*.md`, `pacman_rl/docs/html/**`
   - Nhớ chạy lại `build_search_index.py` trước khi commit để hai file `assets/*-index.js` khớp.
4. **`feat/notebooks`** (hoặc nhánh riêng): `Update Ms. Pacman notebook: PPO run with death_penalty`
   - `notebooks/pacman_rl.ipynb`

Lưu ý: `ppo.py` chứa cả thay đổi `death_penalty` lẫn thay đổi theo dõi số lần chết (và vài chỗ chỉ format lại code). Dùng `git add -p` để tách các đoạn đó vào đúng commit; phần chỉ format lại nên commit riêng (`Reformat ppo.py`) hoặc bỏ đi.

## Trước khi commit

- [ ] `git status` không còn file lạ (không có `runs/`, `.pt`, `.npy`).
- [ ] Code compile được, hoặc đã chạy thử một lần train ngắn.
- [ ] Tài liệu khớp với code nếu đã đổi chữ ký hàm hay tham số config.
- [ ] `python docs/html/check_links.py` báo 0 vấn đề nếu đã sửa docs HTML.
