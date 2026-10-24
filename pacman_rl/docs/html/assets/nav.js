// Mục lục của toàn bộ docs. Thêm trang mới: tạo file .html rồi thêm một dòng vào đây,
// sau đó chạy `python build_search_index.py` (xem contributing/conventions.html).
window.DOCS_VERSION = "0.1.0";
window.DOCS_NAV = [
  { section: "Bắt đầu", pages: [
    { path: "index.html", title: "Tổng quan" },
    { path: "getting-started/installation.html", title: "Cài đặt" },
    { path: "getting-started/quickstart.html", title: "Chạy thử trong 10 phút" }
  ]},
  { section: "Khái niệm", pages: [
    { path: "concepts/environment.html", title: "Môi trường và các loại điểm" },
    { path: "concepts/parallel-envs.html", title: "Env song song và pipeline" },
    { path: "concepts/ppo.html", title: "PPO trong pacman_rl" },
    { path: "concepts/dqn.html", title: "DQN trong pacman_rl" }
  ]},
  { section: "Hướng dẫn", pages: [
    { path: "guides/training.html", title: "Setup và chạy huấn luyện" },
    { path: "guides/hyperparameters.html", title: "Chọn và chỉnh tham số" },
    { path: "guides/evaluation.html", title: "Đánh giá và xem agent chơi" },
    { path: "guides/plotting.html", title: "Vẽ biểu đồ, đọc log" },
    { path: "guides/reward-shaping.html", title: "Reward shaping" }
  ]},
  { section: "Tham chiếu", pages: [
    { path: "reference/ppo-config.html", title: "PPOConfig" },
    { path: "reference/dqn-config.html", title: "DQNConfig" },
    { path: "reference/cli.html", title: "Dòng lệnh (CLI)" },
    { path: "reference/api.html", title: "API Python" },
    { path: "reference/files-and-logs.html", title: "File lưu và log" }
  ]},
  { section: "Đóng góp", pages: [
    { path: "contributing/architecture.html", title: "Kiến trúc mã nguồn" },
    { path: "contributing/add-algorithm.html", title: "Thêm thuật toán mới" },
    { path: "contributing/extend-components.html", title: "Thêm chỉ số, reward, mạng" },
    { path: "contributing/conventions.html", title: "Quy ước, test, viết docs" }
  ]},
  { section: "Khác", pages: [
    { path: "troubleshooting.html", title: "Xử lý sự cố và FAQ" }
  ]}
];
