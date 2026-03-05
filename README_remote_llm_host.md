# Chạy MedAgentSim với LLM host từ xa

Tài liệu này hướng dẫn cách cấu hình và khởi chạy mô phỏng sử dụng **một dịch vụ ngôn ngữ lớn (LLM) được host ở máy chủ từ xa** thay vì chạy model cục bộ (Ollama/vLLM).

Trong dự án ban đầu, `BAgent` mặc định truy vấn server Ollama tại `localhost:11434` hoặc vLLM tại `localhost:8012`. Bây giờ code đã được mở rộng để:

- Nhận **URL của endpoint chat-completions** thông qua tham số, biến môi trường `SERVER_URL` hoặc file cấu hình.
- Gửi thêm header `Authorization: Bearer <token>` nếu cung cấp `SERVER_TOKEN` (hoặc `server_token` khi khởi tạo). 
- Cũng hỗ trợ `OLLAMA_HOST` cho giao tiếp Ollama.

Nhờ vậy bạn có thể trỏ toàn bộ hệ thống sang bất kỳ API OpenAI‑compatible nào (mặc định `/v1/chat/completions`).

---
## 1. Cài đặt bên phía server

1. Trên máy chủ remote (có thể dùng Ollama, vLLM, hoặc dịch vụ LLM thương mại khác) chắc chắn rằng API chat-completions hoạt động tại một URL có thể truy cập từ nơi bạn chạy MedAgentSim. Ví dụ:
   ```text
   https://llmapi.iec-uit.com/v1/chat/completions
   ```
2. Lấy **token API** / khoá bí mật cần thiết để xác thực với dịch vụ. Ví dụ: `sk-llmiec-e90a0e08c8640e7c5995037551a19af5`.

> Nếu server của bạn yêu cầu endpoint hoặc header khác, bạn vẫn có thể dùng `BAgent` vì nó chỉ dùng `requests.post` cơ bản; chỉnh lại trong `_query_server` nếu cần.

---
## 2. Thiết lập môi trường cho MedAgentSim

Bạn có thể bật hai biến môi trường (hoặc thêm vào file cấu hình, xem phần sau):
```powershell
# PowerShell (Windows)
$env:SERVER_URL  = 'https://llmapi.iec-uit.com/v1/chat/completions'
$env:SERVER_TOKEN = 'sk-llmiec-e90a0e08c8640e7c5995037551a19af5'

# bash/zsh
export SERVER_URL='https://llmapi.iec-uit.com/v1/chat/completions'
export SERVER_TOKEN='sk-llmiec-e90a0e08c8640e7c5995037551a19af5'
```

Biến `OLLAMA_HOST` vẫn tồn tại nếu bạn muốn dùng Ollama thay thế (£ mặc định) – ở trường hợp này nó sẽ được bỏ qua bởi vì server vLLM trỏ tới remote và _có thể_ trả 200 từ `/health`.

> **Lưu ý:** các biến trên phải được thiết lập **trước khi** khởi động bất kỳ mô-đun Python nào của MedAgentSim, vì lớp `BAgent` đọc chúng lúc khởi tạo.

---
## 3. (Tùy chọn) Cập nhật file cấu hình `medsim/configs/config_sim.yaml`

Bạn có thể thêm một mục mới để ghi lại đường dẫn và token, và sau đó export chúng khi chương trình khởi chạy. Ví dụ:

```yaml
# mới trong config_sim.yaml
remote_llm:
  url: "https://llmapi.iec-uit.com/v1/chat/completions"
  token: "sk-llmiec-e90a0e08c8640e7c5995037551a19af5"
```

và trong `medsim/simulate/__main__.py` (đã tải config) chèn sớm:

```python
if config.get("remote_llm"):
    os.environ.setdefault("SERVER_URL", config["remote_llm"]["url"])
    os.environ.setdefault("SERVER_TOKEN", config["remote_llm"]["token"])
```

Không nhất thiết phải dùng cấu hình – biến môi trường đơn giản hơn cho mục đích chạy nhanh.

---
## 4. Chạy mô phỏng

Sau khi cài đặt biến môi trường, chạy bình thường như trước:

```powershell
python -m medsim.simulate
# hoặc dùng CLI khác: python run_simulation.py ...
```

`BAgent` sẽ kiểm tra server vLLM tại URL bạn đã cung cấp. Nếu phản hồi `200` ở `/health`, tất cả truy vấn sẽ được gửi tới máy chủ remote với header `Authorization`.

Nếu server không trả lời hoặc có lỗi, `BAgent` sẽ tiếp tục thử Ollama (nếu `OLLAMA_HOST` được bật) hoặc hạ cấp sang mô hình cục bộ.

---
## 5. Ví dụ cụ thể

Giả sử bạn muốn chạy 10 scenario sử dụng host `https://llmapi.iec-uit.com` với token mẫu:

```powershell
$env:SERVER_URL='https://llmapi.iec-uit.com/v1/chat/completions'
$env:SERVER_TOKEN='sk-llmiec-e90a0e08c8640e7c5995037551a19af5'
python -m medsim.simulate
```

Trong log bạn sẽ thấy dòng giống:
```
Using vLLM server: https://llmapi.iec-uit.com/v1/chat/completions
```

Và mọi agent (doctor/patient/etc) sẽ gọi endpoint này qua HTTP.

### 🔧 Gặp lỗi "No server available, loading model locally..."

Nếu bạn thêm `SERVER_URL` và `SERVER_TOKEN` nhưng vẫn thấy thông báo trên và sau đó một traceback kiểu `Unauthorized for url https://huggingface.co/...` thì nghĩa là
- kiểm tra sức khoẻ ban đầu không thành công: `GET <server_url>/health` trả về không phải 200 (404/401/timeout) **và** thử POST nhẹ vào endpoint cũng gặp lỗi.
- trước đây, `BAgent.__init__` sẽ ném `RuntimeError` vì `force_server` được bật (URL được cung cấp); trong phiên bản hiện tại nó sẽ chỉ in cảnh báo và thử gọi thực tế.

Nguyên nhân phổ biến:
* dịch vụ remote không triển khai `/health` endpoint (nhiều API thương mại không có).
* token chưa được gửi (chưa đặt đúng `SERVER_TOKEN` hoặc sai header).
* máy chủ không thể truy cập từ host hiện tại (firewall/NAT). 

**Cách khắc phục:**
1. Đảm bảo địa chỉ và cổng chính xác. Thử `curl` trực tiếp:
   ```bash
   curl -H "Authorization: Bearer $SERVER_TOKEN" \
        -X POST $SERVER_URL \
        -d '{"model":"foo","messages":[{"role":"user","content":"hi"}]}'
   ```
   nếu bạn nhận được phản hồi hợp lệ thì server hoạt động.
2. Nếu server không hỗ trợ `/health`, bạn có thể tạm thời vô hiệu hoá kiểm tra bằng cách xóa biến `SERVER_URL` hoặc thay đổi mã theo hướng dẫn ở phần "mở rộng" phía dưới.  Với phiên bản mã mới, BAgent sẽ hiển thị thông báo như sau và tiếp tục thử:

    ```
    WARNING: Remote server https://... did not pass health check, but SERVER_URL was set; attempting queries anyway. (errors will occur if the host truly is unreachable.)
    ```
3. Kiểm tra token — nếu server trả 401 thì thêm token (hoặc dùng `--server-token`/env).
4. Nếu bạn muốn bỏ qua hoàn toàn kiểm tra sức khoẻ và chấp nhận lỗi lúc gọi, đặt `force_server=False` khi khởi tạo `BAgent` (ví dụ trong code experiment riêng).



---
## 6. Điều khuyên

- Giữ token an toàn; đừng commit vào Git.
- Nếu server cách ly (firewall/NAT), đảm bảo cổng mở và URL chính xác.
- Bạn cũng có thể chạy cùng lúc Ollama nội bộ và remote – `BAgent` sẽ ưu tiên vLLM (remote) trước nếu `/health` trả 200.

---
Bây giờ dự án của bạn đã linh hoạt: chạy cục bộ với Ollama hoặc chuyển sang bất cứ LLM host nào chỉ bằng vài biến môi trường hay một cấu hình đơn giản.