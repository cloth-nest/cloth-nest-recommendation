## Thông tin nhóm
- Phạm Trương Hải Đoàn - MSSV: 20520046
- Mai Phạm Quốc Hưng - MSSV: 20521366

## Sản phẩm
- Mô hình OutfitTransformer để giải quyết bài toán Outfit Compatibility Prediction (Dự đoán độ đồng bộ của trang phục).
- Link GitHub: https://github.com/Doan-Pham/outfit_compatibility_prediction.git
- Bài báo tham khảo: [OutfitTransformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812) của nhóm tác giả Rohan Sarkar, Navaneeth Bodla, Mariya I. Vasileva, Yen-Liang Lin, Anurag Beniwal, Alan Lu và Gerard Medioni

## Hướng dẫn cài đặt
1. Clone repo này về với [Git](https://git-scm.com)
```bash
git clone https://github.com/Doan-Pham/outfit_compatibility_prediction.git
```
2. Tải file [data.zip (2.3GB)](https://drive.google.com/file/d/1696cpHFamwTH9ViyUYlPHCvL0X52Ww16/view?usp=sharing) về cùng thư mục với repo vừa clone và giải nén
3. Cài đặt miniconda3 (Một trình quản lý package và virtual environment cho Python) từ [trang chủ của miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)
4. Sau khi cài đặt, mở Anaconda Prompt (miniconda3)
![Untitled](https://github.com/Doan-Pham/outfit_recommendation/assets/85011400/c0d78c1b-19a8-44bd-ba78-327e13379994)

5. Chạy lệnh sau để tạo Anaconda virtual environment cùng với các package cần thiết.
```bash
conda create -n outfit_recommendation --file requirements.txt
```

## Hướng dẫn chạy source
1. Mở Anaconda Prompt và di chuyển đến thư mục chứa source:
```bash
cd path/to/your/repo/outfit_recommendation
```
2. Kích hoạt môi trường đã tạo ở bước cài đặt:
```bash
conda activate outfit_recommendation
```
Lúc này khi môi trường được kích hoạt, chương trình đã có đủ các package cần thiết để chạy.

3. Chạy lệnh sau để tiến hành train model:
```bash
python main.py
```
Mặc định, model sẽ không được train trên toàn bộ tập dữ liệu khi nhập lệnh ở trên mà chỉ train với một số lượng dữ liệu rất nhỏ nhằm minh họa. Để train với toàn bộ tập dữ liệu, gõ lệnh sau:
```bash
python main.py --run_real 1
```
Kết quả train của mỗi vòng epoch được lưu trong thư mục `checkpoint/disjoint`, trong đó các file với định dạng `checkpoint_0.pt`, `checkpoint_1.pt` lưu kết quả train của từng epoch tương ứng, còn file `best_state.pt` lưu kết quả train với chỉ số chính xác cao nhất. 

Có thể nhập lệnh sau để biết các argument có thể sử dụng với lệnh `python main.py` ở trên:
```bash
python main.py --help
```
Kết quả khi chạy lệnh `python main.py --help`
```bash
usage: main.py [-h] [--datazip DATAZIP] [--run_real RUN_REAL]
               [--log_level LOG_LEVEL] [--datadir DATADIR]
               [--checkpoint_dir CHECKPOINT_DIR] [--batch_size BATCH_SIZE]
               [--polyvore_split POLYVORE_SPLIT] [--epochs EPOCHS]

options:
  -h, --help            show this help message and exit
  --datazip DATAZIP     Path to input data zip file
  --run_real RUN_REAL   0 = train with few data to see model run; 1 = train with    
                        whole dataset. Default is 0
  --log_level LOG_LEVEL
                        0 = Print >= warnings, 1 = print >= info, 2 = print all     
  --datadir DATADIR     Path to data directory
  --checkpoint_dir CHECKPOINT_DIR
                        Path to the directory to save checkpoints
  --batch_size BATCH_SIZE
                        Batch size in training, default is 50
  --polyvore_split POLYVORE_SPLIT
                        The split of the polyvore data (disjoint or nondisjoint)    
  --epochs EPOCHS       Number of epochs to train for (default: 10)
```
Ví dụ sử dụng lệnh `python main.py` để huấn luyện model trong 15 epoch với số lượng mẫu mỗi đợt huấn luyện là 30, đồng thời lấy dữ liệu từ file data.zip thay vì lấy từ thư mục đã được giải nén và tiến hành train trên toàn bộ tập dữ liệu.
```bash
python main.py --epochs 15 --batch_size 30 --datazip data.zip --run_real 1
```
**Lưu ý**: Train với lượng dữ liệu lớn sẽ dể làm đứng máy hoặc crash IDE vì thiếu bộ nhớ => Có thể giảm bacth_size xuống hoặc không để --run_real là 1 (Hoặc sử dụng máy với cấu hình tốt hơn).

4. Khi đã train xong model với ít nhất 1 epoch (hoặc bạn có thể tải [model đã được train ở đây (240 MB)](https://drive.google.com/file/d/1GnA3LGX_bTvWn08k0SPaEzNaxHSSljMn/view?usp=sharing) và giải nén vào thư mục chứa repo), có thể chạy lệnh sau để mở app demo. App này sẽ sử dụng tham số trong file `best_state.pt` để áp dụng cho model và đưa ra dự đoán:
```bash
streamlit run demo_app.py
```
