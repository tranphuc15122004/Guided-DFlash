# **Mô hình hóa bài toán $M = (S , A , P , R , \gamma)$**

## Problem setup

Trong mô hình diffusion speculative decoding Dflash, việc áp dụng cơ chế contrastive decoding đang bị giới hạn hiệu quả bởi việc lựa chọn hệ số alpha (pos - alpha * neg). Cố định một alpha cho toàn bộ kịch bản cho thấy việc không hiệu quả, cần có một cơ chế điều khiển hệ số alpha động cho cơ chế CD trong từng decode block. Ý tưởng hiện tại ta sẽ train một model RL nhằm điều chỉnh alpha theo output của mô hình drafter

- block size = 16 → 15 token cần diffusion mỗi lần

## State $S$

Tập trạng thái của model , tại mỗi thời điểm drafter đưa ra kết quả sẽ là một lần alpha agent sẽ đưa ra quyết định.

- Positive sample (32 top rank token)
    - log(logit)
    - log(softmax(logit))
    - softmax(logit)
    - các thống kê (entropy , top1 margin, top-k mass ,. .. )
- Negative sample
    - log(logit)
    - log(softmax(logit))
    - softmax(logit)
    - các thống kê (entropy, top1 margin, top-k mass , …)
- Cross sample features
    - hiệu của 2 sample (pos - neg)
    - KL divergence
- Position: (block position , vị trí tuyệt đối của từng token trong câu) (normalize)
- alpha tại decode block trước đó $\alpha_{t-1}$

## Action $A$

Một hành động của model là để xuất ra một hệ số alpha phù hợp tại mỗi thời điểm.

Tại môi token sẽ đưa ra 3 hệ số alpha. 

Phạm vi áp dụng của alpha sẽ là mỗi alpha sẽ được áp dụng cho 10 top token. Ví dụ top 10 token có rank cao nhất trong positive sample logit sẽ được áp dụng alpha [0], từ top 11 đến top 20 sẽ được áp dụng alpha [1] , …

$$
a_t  \in \R^{(B-1) \times 3} : \text{cho mỗi bucket 10 top token}
$$

Tại sao phải sử dụng 3 hệ số alpha thay vì dùng từng alpha cho từng token ?

Việc sử dụng 3 token này sẽ làm loose hơn yêu cầu học của model, thay vì phải học cách tìm được vị trí của target token ở đầu trong dữ liệu đầu vào là phân phối của logit hệ thống (khá khó)

## Reward $R$

Reward 1 sẽ giúp model học được việc đây rank lên, cũng như tránh model làm xấu đi rank của target

$$
r_1 = \sum_{i = 1}^{B-1} \Delta_{target\ rank} * \exp(- \frac{i-1 }{\gamma})
$$

Reward 2 sẽ thưởng thêm nếu model đẩy được target lên top1, phần thưởng này nên lớn hơn so với reward 1 để model tập trung đẩy target token lên top 1 để được fix được draft model thay vì chỉ đẩy lên một cách dàn trải như reward 1

$$
r_2 = \sum_{i =1}^{B-1} 2 \times \exp(- \frac{i-1}{\gamma}) \times \text{target rank len top 1}
$$

Reward 3 thì ưu tiên làm tăng acclength của model, mục tiêu chính của bài toán. Bằng cách định nghĩa này thì model sẽ phải học rằng model ưu tiên việc không làm tệ đi trước accpetance trung bình trước khi cải thiện nó.

$$
r_3 = \max(\Delta  acc\_length , 0 ) - \lambda \max(-\Delta acc\_length , 0 ) 
$$

Hàm reward tổng hợp :

$$
R = w_1r_1 + w_2r_2  +w_3r_3
$$

Thử bắt đầu bằng $w_1  = 0.1 , w_2 = 0.1 , w_3 = 1.0 , \lambda = 3.0 , \beta = 0.02 , \gamma  =7$ 

## Episode

một episode là một quá trình decode ra toàn bộ câu trả lời cho một câu hỏi 

(s , a ,s a, ….)

Thu thập dữ liệu → train ngẫu nhiê

## Transition $P$

sau quá trình drafting của draft model, kết quả thu được là positive và negative sample có kích thước $R^{(B-1) \times dict \ size}$

mô hình alpha RL (nhẹ) sẽ được sử dụng để đưa ra một hệ số alpha adaptive cho cơ chế contrastive hiện tại 

Target model sẽ verify logit đầu ra của cơ chế contrastive , trả về acceptance length và fixed token tại vị trí reject