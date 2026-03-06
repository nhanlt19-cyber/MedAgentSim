# Scenario Auto-Increment Feature

## Overview
Tính năng này cho phép bạn chạy simulation nhiều lần mà không lo về việc **ghi đè lên dữ liệu cũ**. Mỗi lần chạy, các scenario sẽ được lưu với ID tiếp theo tự động.

## How It Works / Cách Hoạt Động

### Trước (Old Behavior)
```
Lần chạy 1:
- output/scenario_0/
- output/scenario_1/
- output/scenario_2/

Lần chạy 2: (GHI ĐÈ Lên cũ ❌)
- output/scenario_0/  ← DỮ LIỆU CŨ BỊ MẤT
- output/scenario_1/  ← DỮ LIỆU CŨ BỊ MẤT
- output/scenario_2/  ← DỮ LIỆU CŨ BỊ MẤT
```

### Sau (New Behavior)
```
Lần chạy 1:
- output/scenario_0/
- output/scenario_1/
- output/scenario_2/

Lần chạy 2: (TỰ ĐỘNG TĂNG ID ✓)
- output/scenario_0/  ← LƯU GỮ
- output/scenario_1/  ← LƯU GỮ
- output/scenario_2/  ← LƯU GỮ
- output/scenario_3/  ← KẾT QUẢ CHẠY MỚI
- output/scenario_4/  ← KẾT QUẢ CHẠY MỚI
- output/scenario_5/  ← KẾT QUẢ CHẠY MỚI

Lần chạy 3: (TIẾP TỤC TĂNG ✓)
- output/scenario_0-5/  ← LƯU GỮ
- output/scenario_6/    ← KẾT QUẢ CHẠY MỚI
- output/scenario_7/    ← KẾT QUẢ CHẠY MỚI
- output/scenario_8/    ← KẾT QUẢ CHẠY MỚI
```

## Implementation Details / Chi Tiết Cài Đặt

### Changes Made
1. **Thêm hàm `find_next_available_scenario_id()`** trong `medsim/run.py`
   - Quét folder `output/` để tìm ID scenario cao nhất
   - Trả về `max_id + 1` để tránh trùng

2. **Sửa hàm `run_simulation()`**
   - Gọi `find_next_available_scenario_id()` trước khi chạy
   - Tính `output_scenario_id = start_id + scenario_index`
   - Truyền `output_scenario_id` vào `run_interaction_loop()`

3. **Sửa hàm `run_simulation_idx()`** (chạy từng scenario riêng lẻ)
   - Tương tự như `run_simulation()`
   - Hỗ trợ chạy từng scenario lần lượt mà không ghi đè

4. **Sửa hàm `run_interaction_loop()`**
   - Thêm tham số `output_scenario_id`
   - Sử dụng `output_scenario_id` thay vì `scenario_id` khi lưu file
   - `scenario_id` vẫn được dùng để lấy dữ liệu từ dataset

## Usage Examples / Ví Dụ Sử Dụng

### Example 1: Chạy simulation bình thường
```bash
# Lần chạy 1
python -m medsim.run --config medsim/configs/config.yaml

# Kết quả:
# output/scenario_0/dialogue_history.json
# output/scenario_1/dialogue_history.json
# ... (tùy số lượng scenario trong config)

# Lần chạy 2 (không lo ghi đè)
python -m medsim.run --config medsim/configs/config.yaml

# Kết quả: 
# output/scenario_0-N/ (cũ - được giữ)
# output/scenario_N+1-N+1+M/ (mới - tự động tăng ID)
```

### Example 2: Chạy từng scenario riêng lẻ
```bash
# Lần 1: chạy scenario 0 từ dataset
python examples/run_backend.py --scenario_id 0

# Kết quả: output/scenario_0/dialogue_history.json

# Lần 2: chạy scenario 0 từ dataset lại (không lo ghi đè)
python examples/run_backend.py --scenario_id 0

# Kết quả: 
# output/scenario_0/ (cũ - được giữ)
# output/scenario_1/ (mới)
```

## Testing

Để kiểm tra tính năng hoạt động đúng, chạy test script:

```bash
python test_scenario_increment.py
```

Output mong đợi:
```
✓ All tests passed!
SUCCESS: Auto-increment scenario ID is working correctly!
```

## Files Modified / File Được Sửa

- `medsim/run.py` (chính)
  - Thêm `find_next_available_scenario_id()` function
  - Sửa `run_simulation()` function
  - Sửa `run_simulation_idx()` function
  - Sửa `run_interaction_loop()` function signature

## Backward Compatibility / Tương Thích Ngược
✓ **Hoàn toàn tương thích ngược** - không cần thay đổi code cũ
- Các hàm vẫn hoạt động giống như trước nếu `output_scenario_id=None`
- Sử dụng mặc định từ `scenario_id` nếu không chỉ định

## Important Notes / Lưu Ý Quan Trọng

1. **Output Format không thay đổi**
   - Vẫn lưu tại: `output/scenario_<ID>/dialogue_history.json`
   - Format JSON vẫn giống như cũ

2. **Multi-run Safety**
   - A/A testing: có thể chạy cùng scenario lần lượt không lo mất dữ liệu
   - Dữ liệu cũ được bảo vệ tự động

3. **Logging**
   - Thêm log: `Starting output scenario ID: <ID>`
   - Log mỗi lần tìm ID tiếp theo

## Troubleshooting / Khắc Phục Sự Cố

### Problem: Vẫn thấy ghi đè dữ liệu
**Solution**: 
- Đảm bảo bạn đã cập nhật `medsim/run.py`
- Kiểm tra log xem có dòng `Starting output scenario ID` không
- Chạy `test_scenario_increment.py` để verify

### Problem: ID không tăng từ 0
**Solution**:
- Điều này là bình thường nếu đã chạy trước đó
- Hàm sẽ tìm max ID hiện tại và tăng từ đó

### Problem: Không thể đọc folder output
**Solution**:
- Đảm bảo folder `output/` tồn tại
- Hàm sẽ tự tạo nếu không tồn tại
- Kiểm tra quyền truy cập file

## Questions / Câu Hỏi?

Nếu có vấn đề, hãy:
1. Chạy `test_scenario_increment.py` để kiểm tra
2. Xem log file `simulation.log`
3. Kiểm tra thư mục `output/` xem scenario folder

---
**Last Updated**: March 6, 2026
**Feature Status**: ✓ Tested and Working
