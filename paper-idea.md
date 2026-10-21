# Ý tưởng bài báo: Gauge Centring cho XAI dựa trên phân rã DC

**Nguồn trích dẫn**

- **[P1]** *The Second Exponential Belongs to the Rule: Removing Double-Exponential Cancellation in DC Deep Networks by Gauge Centring* — [Paper1.pdf](../papers/Paper1.pdf)
- **[DCD]** *DC Computational Calculus for Deep Networks* — [dc_deep_networks_full_paper.pdf](../papers/dc_deep_networks_full_paper.pdf)
- **[NOTE]** Ghi chú rà soát §4.6 "Bùng nổ triệt tiêu", Chương 4 — [NOTE_ch4_cancellation_review.pdf](../papers/NOTE_ch4_cancellation_review.pdf)
- **[HM]** Zimmermann & Loho, *Hidden Monotonicity: Explaining Deep Neural Networks via their DC Decomposition*, CVPR 2026, arXiv:2601.07700 — [hidden-monotonicity-explain-dnn-by-dc.pdf](../papers/hidden-monotonicity-explain-dnn-by-dc.pdf)

**Ký hiệu chung**

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $f$ | mạng (hoặc một khối/tầng) cần giải thích, đầu ra vô hướng (logit của lớp $y$) |
| $(P,Q)$, $(g,h)$ | cặp DC, $f = P-Q = g-h$; [P1] dùng $(P,Q)$, [HM] dùng $(g,h)$ |
| $x^\star$ | điểm cần giải thích (ảnh đầu vào hoặc kích hoạt tầng) |
| $u$, $b$ | đơn vị làm tròn và số bit mantissa (float16/32/64: $b = 11/24/53$) |
| $\chi_{\mathrm{DC}}$ | số điều kiện triệt tiêu phép trừ |
| $C$ | cận độ cong, cũng là hệ số của gauge bậc hai |
| $\Omega_\rho(x^\star)$ | miền tin cậy bán kính $\rho$ quanh $x^\star$ |

---

## 1. Introduction — Giới thiệu và động lực từ Hidden Monotonicity

### 1.1 Bối cảnh: XAI dựa trên tính đơn điệu / lồi

- Mạng đơn điệu (mọi trọng số không âm) và mạng lồi theo đầu vào (ICNN — mọi trọng số trừ tầng đầu không âm) có tính tự giải thích cao: không xảy ra triệt tiêu dấu giữa trọng số âm/dương, nên tách rõ được đặc trưng có mặt (*factual*) và đặc trưng khuyết thiếu (*counterfactual*).
- Cơ sở hình thức [HM, Thm 2]: nếu $g$ đơn điệu và tuyến tính từng khúc liên tục (CPWL) thì $\nabla g \ge 0$ — mọi gradient đạt được đều không âm.
- Nhược điểm: ép trọng số không âm ngay từ lúc huấn luyện làm giảm mạnh khả năng biểu diễn.

### 1.2 Bước tiến của Hidden Monotonicity và rào cản số học

- [HM] tách **bất kỳ** mạng ReLU đã huấn luyện nào thành hiệu hai luồng đơn điệu, lồi $f = g - h$, dựa trên đồng nhất thức
  $$\operatorname{ReLU}(a-b) = \max\{a-b,\,0\} = \max\{a,b\} - b,$$
  rồi xây dựng SplitGrad, SplitCAM, SplitLRP.
- Chính [HM] gọi hiện tượng gặp phải là một *"inherent blowup of the weights"* trong thủ tục tách. Khi không ổn định hoá lượt đi, kích hoạt của luồng tách ở các tầng cuối VGG16 đạt $\sim 10^{18}$ và sai số $\sim 10^{9}$, dù đã hiệu chỉnh kích hoạt theo mạng gốc ở từng tầng; SplitLRP cho bản đồ gần như trắng toàn bộ [HM, App. C.2].

### 1.3 Hạn chế của các giải pháp vá lỗi thực nghiệm (ad-hoc stabilization)

[HM] cứu phép phân tách bằng hai thủ thuật:

1. **Lượt đi (forward scaling/shifting).** Khi $\lvert a\rvert > \Theta$, nhân cả tensor kích hoạt với $\theta$ ($\theta = 0.1$, $\Theta = 10$). Mọi phép tính dùng `torch.float64`, kèm tuỳ chọn hiệu chỉnh theo cache của mạng gốc để giữ **bất biến lượt đi**
   $$a^{(l,+)} - a^{(l,-)} = a^{(l)}.$$
2. **Lượt về (backward $\alpha$-shifting).** Ở mỗi tầng, trừ đi một bội $\alpha^{(l)}$ của tổng độ nhạy hai nhánh [HM, §3.3, App. F.3]:
   $$\delta^{(l,\pm,g)}_{\text{shift}} = \delta^{(l,\pm,g)}_{\text{inter}} - \alpha^{(l)}\big(\delta^{(l,+,g)}_{\text{inter}} + \delta^{(l,-,g)}_{\text{inter}}\big),$$
   tương tự cho $h$. Vùng "đáng quan tâm" là $0 < \alpha < 0.5$.

Hệ quả:

- tốn bộ nhớ lưu cache toàn mạng và phụ thuộc siêu tham số $\alpha$ (chất lượng dao động mạnh theo $\alpha$);
- chậm hơn LayerCAM khoảng $27$–$29\times$ [HM, Tab. 7];
- với nút nhân (gated unit, attention), [HM, App. E] mới có lý thuyết tách cho *Multiplicative Maxout network*; việc áp dụng SplitCAM/SplitLRP cho Transformer được để ngỏ. [P1] cho thấy biến thể "naive + multiplicative shifting (HM)" sụp về sai số tương đối $1$ ở độ sâu $L \approx 7$ (mạng cổng) và $N \approx 4$ khối (Transformer linear attention).

### 1.4 Câu hỏi nghiên cứu và đóng góp

**Câu hỏi trung tâm [P1]:** sự bùng nổ thuộc về **hàm** được biểu diễn hay về **quy tắc** biểu diễn nó?

**Phạm vi phát biểu** [P1, §1.1], [NOTE, §4]:

- Đây *không* phải khuyết tật của phân rã DC nói chung — cùng một hàm có phân rã DC điều kiện tốt. Đây là khuyết tật của *một cách xây dựng*: lan truyền đệ quy với quy tắc nhân bình phương.
- Đây cũng *không* phải khuyết tật của mạng nơ-ron nói chung. Backprop cũng xử lý đồng thời mọi tầng mà không nổ, vì nó **không bao giờ dựng $P$ và $Q$**. Cái giá đến từ việc *mang theo một biểu diễn DC*. Với XAI, lý do để mang theo $(g,h)$ là ta cần $\nabla g$ và $\nabla h$ **riêng rẽ**.
- Quy tắc nhân bình phương là một *lựa chọn hiện thực hoá* (để cặp đầu ra cắm thẳng được vào nút kế tiếp), không phải điều bắt buộc. Nó chỉ cần khi **cả hai thừa số đều biến thiên** — $x\cdot w$ với $w$ cố định là tuyến tính.
- Thuật ngữ: gọi là **bùng nổ triệt tiêu** (cancellation blow-up), không gọi là "bùng nổ đạo hàm". Hai hiện tượng được chứng minh là bậc đa thức ($2^\ell$) và độ lớn giá trị (mũ kép) của hai thành phần [NOTE, §1].

**Đóng góp dự kiến**

1. **Chẩn đoán toán học chính xác.** Định lượng hiện tượng bằng $\chi_{\mathrm{DC}}$; chứng minh ranh giới sắc nét giữa chế độ mũ đơn (nhân với trọng số cố định — CNN/MLP) và chế độ mũ kép, ví dụ $\chi_{\mathrm{DC}} \ge 2^{2^{L-1}}$, khi có nút nhân tự củng cố (gated unit, $QK^\top$, $SV$).
2. **Sự bất lực của phép co giãn.** $\chi_{\mathrm{DC}}$ bất biến dưới $(P,Q)\mapsto(\theta P,\theta Q)$, nên co giãn lượt đi chỉ chống tràn số. Phép dịch $\alpha$ lượt về cần được đọc lại như một **phép chọn gauge thực nghiệm** (xem §3.3).
3. **Gauge centring.** Sự bùng nổ không thuộc về $f$ mà do gauge lồi ngầm trong quy tắc nhân được đặt tâm tại gốc. Dịch tâm gauge bậc hai về đúng $x^\star$ cho $\chi_{\mathrm{DC}}(x^\star) = 1$, khôi phục giá trị và đạo hàm đến độ chính xác máy mà không cần cache từng tầng hay tham số $\alpha$.
4. **Ứng dụng XAI.** Các phương pháp giải thích dựa trên gauge centring, so sánh với [HM] về độ chính xác số học, tốc độ và các chỉ số Quantus — **cần giải quyết các vấn đề ở §4.3 trước**.

---

## 2. Preliminaries: Giải tích phân rã DC và tự do gauge

*Nguồn: [P1, §2.1–2.2], [DCD, §3–4, §6.1], [HM, §2–3].*

### 2.1 Cặp DC không âm (NDC pair) và phép tách mạng

**Định nghĩa (NDC pair).** $(P,Q)$ là cặp NDC của $u$ trên $\Omega$ nếu $P, Q : \Omega \to [0,\infty)$ lồi và $u = P - Q$.

**Bổ đề 1 (quy tắc nguyên thuỷ)** [P1, Lem. 1], [DCD, Lem. 4.1]. Với $u = P-Q$, $v = R-S$, trong đó $P,Q,R,S \ge 0$ lồi:
$$u+v = (P+R)-(Q+S), \qquad -u = Q-P,$$
$$uv = \tfrac12\big[(P+R)^2+(Q+S)^2\big] - \tfrac12\big[(P+S)^2+(Q+R)^2\big].$$
Mọi thành phần đều lồi và không âm, vì $t\mapsto t^2$ lồi, tăng trên $[0,\infty)$. Phép cộng và phép đổi dấu không làm tăng độ lớn; **phép nhân bình phương độ lớn thành phần** — đây là nơi chi phí xuất hiện.

**Nâng kích hoạt chuẩn tắc (canonical lift)** [P1, (4)], [DCD, Def. 4.2, Thm 5.1]:
$$\Phi_\sigma(a,b) := \sigma(a-b) + b, \qquad \sigma(P-Q) = \Phi_\sigma(P,Q) - Q.$$
$\Phi_\sigma$ lồi đồng thời và không giảm theo từng toạ độ $\iff$ $\sigma$ lồi, không giảm và 1-Lipschitz. Ví dụ:

- ReLU: $\Phi(a,b) = \max\{a,b\}$, tức $(P,Q) \mapsto (\max\{P,Q\},\, Q)$;
- Softplus $\sigma_\tau(t) = \tau\log(1+e^{t/\tau})$: $\Phi(a,b) = \tau\log\big(e^{a/\tau}+e^{b/\tau}\big)$.

**Phép tách mạng ReLU** [HM, §3.1]. Tách $W^{(l)} = W^{(l,+)} - W^{(l,-)}$ với $W^{(l,\pm)} \ge 0$, và $x = a^{(0,+)} - a^{(0,-)}$:
$$z^{(l,+)} = W^{(l,+)}a^{(l-1,+)} + W^{(l,-)}a^{(l-1,-)}, \qquad z^{(l,-)} = W^{(l,-)}a^{(l-1,+)} + W^{(l,+)}a^{(l-1,-)},$$
$$a^{(l,+)} = \max\{z^{(l,+)},\, z^{(l,-)}\}, \qquad a^{(l,-)} = z^{(l,-)}.$$

- **Định lý 1 [HM]:** $f^{(l)}(x^+ - x^-) = g^{(l)}(x^+,x^-) - h^{(l)}(x^+,x^-)$, với $g, h$ đơn điệu, lồi, CPWL.
- Max pooling [HM, §3.4]: $\max_{j}(a_j^+ - a_j^-) = \max_{j}\big\{a_j^+ + \sum_{i\ne j}a_i^-\big\} - \sum_i a_i^-$.
- Nhận xét (heuristic): *tổng* hai luồng lan truyền qua $\lvert W^{(l)}\rvert = W^{(l,+)} + W^{(l,-)}$, trong khi *hiệu* lan truyền qua $W^{(l)}$. Độ lớn thành phần cỡ $\big\|\prod_\ell \lvert W^{(\ell)}\rvert\big\|$, còn giá trị $f$ cỡ $\big\|\prod_\ell W^{(\ell)}\big\|$; tỉ số giữa hai đại lượng này điều khiển $\chi_{\mathrm{DC}}$.

### 2.2 Tự do gauge (gauge freedom)

Phân rã DC không duy nhất: với mọi hàm lồi khả vi $C$ (một *convex gauge*),
$$(g,h) \mapsto (g+C,\ h+C) \quad\text{giữ nguyên } f = g-h.$$

**Mệnh đề 4 [P1] / Mệnh đề 6.1 [DCD] (gauge chung = số hạng Bregman).** Gọi majorant DCA tại $x$ là $M_{G,H}(z\mid x) = G(z) - H(x) - \langle\nabla H(x),\, z-x\rangle$. Khi đó
$$M_{G+C,\,H+C}(z\mid x) = M_{G,H}(z\mid x) + D_C(z,x), \qquad D_C(z,x) = C(z) - C(x) - \langle\nabla C(x),\, z-x\rangle.$$
Với $C(z) = \tfrac12 z^\top M z$ thì $D_C(z,x) = \tfrac12\|z-x\|_M^2$. Như vậy, chọn phân rã DC = chọn gauge lồi = chọn một regulariser proximal cho DCA. Gauge affine (hằng hoặc tuyến tính) có $D_C \equiv 0$.

**Phép dịch lượt đi của [HM] là một gauge hằng** [HM, Thm 6]. Dịch hai luồng ở tầng $l$ cùng một lượng $c$:
$$g^{(l)}\big(a^{(+)}+c,\ a^{(-)}+c\big) = g^{(l)}\big(a^{(+)},a^{(-)}\big) + W^{[l,\to]}_{\text{abs}}\,c, \qquad W^{[l,\to]}_{\text{abs}} := \prod_{j=L,\dots,l+1}\big\lvert W^{(j)}\big\rvert,$$
và tương tự cho $h$ — tức là cộng cùng một hằng vào cả $g$ và $h$.

---

## 3. Giải phẫu sự bất ổn định số học trong phân rã DC mạng sâu

*Nguồn: [P1, §2.2–2.3], [DCD, §6], [NOTE], [HM, §3.2–3.3, App. C, F].*

### 3.1 Số điều kiện triệt tiêu phép trừ $\chi_{\mathrm{DC}}$

**Định nghĩa 2 [P1].** Với cặp NDC $(P,Q)$ và $P(x) \ne Q(x)$:
$$\chi_{\mathrm{DC}}(x) := \frac{P(x)+Q(x)}{\lvert P(x)-Q(x)\rvert} \ \ge\ 1, \qquad \text{khi } f>0:\ \ \chi_{\mathrm{DC}}(x) = 1 + \frac{2Q(x)}{f(x)}.$$
Điều kiện hoá chính là độ lớn của thành phần âm (tức gauge) đo so với hàm.

**Bổ đề 3 (cận triệt tiêu) [P1].** Nếu $\hat P = P(1+\delta_1)$, $\hat Q = Q(1+\delta_2)$ với $\lvert\delta_i\rvert \le u$, thì
$$\frac{\big\lvert(\hat P-\hat Q)-(P-Q)\big\rvert}{\lvert P-Q\rvert} \ \le\ u\,\frac{P+Q}{\lvert P-Q\rvert} = u\,\chi_{\mathrm{DC}},$$
và cận này đạt được khi $\delta_1 = -\delta_2 = u$. $\chi_{\mathrm{DC}}$ không phải đại lượng thay thế cho hư hại số học — nó *chính là* hư hại. Khi $\chi_{\mathrm{DC}} \gtrsim 1/u$, mọi chữ số có nghĩa của $P - Q$ bị xoá (catastrophic cancellation). Áp dụng tương tự cho từng thành phần gradient $\partial_i P - \partial_i Q$.

**Hai tính chất dùng cho phần phê phán:**

- Co giãn: $\chi(\theta P,\theta Q) = \chi(P,Q)$ với mọi $\theta > 0$.
- Dịch hằng: $\chi(P+c,\,Q+c) = \chi(P,Q) + 2c/\lvert f\rvert$. $\chi$ chỉ giảm khi $c<0$, tức phải trừ đi một hằng gần bằng $\min\{P,Q\}$. Nếu $P, Q$ đã khổng lồ thì chính phép trừ này lại triệt tiêu, nên phải dịch *ở từng tầng, trước khi độ lớn kịp tăng*. Muốn chọn đúng $c$ thì phải biết trước $a^{(l)}$ của mạng gốc (cache).

### 3.2 Định lý phân loại hai chế độ bùng nổ theo độ sâu

**Chuỗi mẫu** [P1, (6)], [DCD, (21)]: $a_0 = 1$, $a_\ell = \operatorname{ReLU}(w_\ell a_{\ell-1})$, $\ell = 1,\dots,L$. Khởi tạo mỗi trọng số bằng cặp $(\max(w_\ell,0),\, \max(-w_\ell,0))$, nhân bằng quy tắc (3), rồi áp ReLU lift.

**Định lý 5 (triệt tiêu mũ kép) [P1].** Tại $w_1 = \dots = w_L = 1$ thì $a_\ell = 1$ với mọi $\ell$, nhưng
$$P_\ell = q_\ell + 1,\quad Q_\ell = q_\ell,\quad q_0 = 0,\quad q_{\ell+1} = (q_\ell+1)^2,\quad q_\ell \ge 2^{2^{\ell-1}}\ (\ell\ge 2),\quad \chi_{\mathrm{DC}}(\ell) = 2q_\ell + 1.$$

| $\ell$ | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- |
| $\chi_{\mathrm{DC}}(\ell)$ | 9 | 51 | 1353 | $9.17\times10^{5}$ | $4.20\times10^{11}$ | $8.83\times10^{22}$ |

Hai thành phần vượt mantissa binary64 gần $\ell = 7$ và tràn số gần $\ell = 11$.

- **Hệ quả 6 (ngân sách bit).** Để chứng nhận độ chính xác tương đối $\delta$ cần $b \ge \log_2\!\big(\chi_{\mathrm{DC}}(\ell)/\delta\big) \ge 2^{\ell-1} + \log_2(1/\delta)$.
- **Mệnh đề 7 (bùng nổ bậc).** Bậc đa thức của hai thành phần là $d_\ell = 2^\ell$, trong khi $a_\ell = \prod_{j\le\ell} w_j$ chỉ có bậc $\ell$.

**Mệnh đề 8 + Định lý 9 (luật tăng trưởng tổng quát) [P1].** Một nút nhân kết hợp cặp đang chạy $(P_\ell, Q_\ell)$ với cặp nhân tử $(R_\ell, S_\ell) \ge 0$. Đặt $M_\ell = \max\{P_\ell, Q_\ell\}$ và $N_\ell = \max\{R_\ell, S_\ell\}$.

- Tích mức giá trị ($P_{\ell+1} = P_\ell R_\ell + Q_\ell S_\ell$, $Q_{\ell+1} = P_\ell S_\ell + Q_\ell R_\ell$): $\tfrac12 M_\ell N_\ell \le M_{\ell+1} \le 2M_\ell N_\ell$.
- Quy tắc bình phương (3): $M_{\ell+1} \ge \tfrac12 M_\ell^2$.
- **Nút tự củng cố** (dùng quy tắc bình phương, hoặc $N_\ell \ge \gamma M_\ell$ — hai toán hạng đều là cặp tích luỹ cỡ tương đương): $M_{\ell+1} \ge c\,M_\ell^2$ với $c = \tfrac12$ (bình phương) hoặc $c = \gamma/2$ (song tuyến). Khi $cM_{\ell_0} > 1$:
  $$M_L \ \ge\ \tfrac1c\,(cM_{\ell_0})^{2^{L-\ell_0}} \qquad \text{(mũ kép).}$$
- **Nhân tử bị chặn** $N_\ell \le \bar N$: $M_L \le M_0\prod_{\ell<L} 2N_\ell \le M_0(2\bar N)^L$ (mũ đơn).

**Chế độ 1 — mũ đơn (CNN/MLP trọng số cố định, đúng thiết lập của [HM]).** Khi giải thích theo đầu vào $x$, trọng số là hằng nên mỗi tầng là một nút mức giá trị với nhân tử bị chặn. Độ lớn thành phần tăng như $\big\|\prod_\ell\lvert W^{(\ell)}\rvert\big\|$ [HM, (29)] — dùng $\lvert W\rvert$, không phải $W$. Đây là lời giải thích lý thuyết cho mức $10^{18}$ ở các tầng cuối VGG16. Ước lượng bậc: một phép trừ hai số cỡ $10^{18}$ trong float64 có sai số tuyệt đối $\sim 10^{18}\cdot 2^{-53} \approx 10^{2}$. Mức sai số $\sim 10^9$ mà [HM] báo cáo phản ánh sự tích luỹ qua nhiều tầng và tổng nhiều số hạng — cần kiểm chứng lại ở thực nghiệm nhóm A.

**Chế độ 2 — mũ kép (mạng cổng, Transformer).** Các nút nhân tự củng cố gồm gated unit $g(x)\odot h(x)$, multiplicative Maxout, điểm attention $QK^\top$ và phép tổng hợp $SV$ (hai nút nhân mỗi khối) [P1, Ex. 11]. Số liệu thực nghiệm của [P1]:

- mạng cổng: sụp ở $L \approx 7$;
- Transformer linear attention: $\chi_{\mathrm{DC}} = 4.2\times10^{1},\ 2.0\times10^{3},\ 3.0\times10^{11}$ sau 1–3 khối và tràn số ở khối 4, trong khi $\lvert f\rvert = O(1)$;
- với $b$ bit mantissa, độ sâu chứng nhận được chỉ là $\Theta(\log b)$ [P1, Cor. 13]; thực đo với float16/32/64 lần lượt là 3/5/6 tầng [P1, Tab. 8].

**Ba độ phức tạp của cùng một hàm (chuỗi độ sâu $L$)** [P1, Tab. 4]:

| Đại lượng | Hàm thật | Biểu diễn DC naive | Gauge-centred (đề xuất) |
| --- | --- | --- | --- |
| Bậc đa thức | $L$ | $2^L$ | $L$ |
| Độ lớn thành phần $M_L$ | $O(1)$ | $2^{2^{L-1}}$ | mũ đơn; $O(1)$ tại $x^\star$ |
| $\chi_{\mathrm{DC}}$ | $1$ | $2^{2^{L-1}}$ | $1$ tại tâm |
| Số bit để chứng nhận | $O(\log\frac1\delta)$ | $\ge 2^{L-1} + \log_2\frac1\delta$ | $O(L + \log_2\frac1\delta)$ |
| Độ sâu chứng nhận với $b$ bit | — | $\Theta(\log b)$ | $\Theta(b)$ |

### 3.3 Vì sao kỹ thuật ổn định của Hidden Monotonicity chưa giải quyết gốc rễ

**Co giãn lượt đi.** $(P,Q)\mapsto(\theta P,\theta Q)$ giữ nguyên $\chi_{\mathrm{DC}}$: nó chỉ dời số mũ chứ không cứu được mantissa. Trong [P1, Fig. 2–3, Tab. 5], đường "naive" (A) và "naive + multiplicative shifting (HM)" (B) trùng nhau và cùng sụp về sai số tương đối $1$.

**Ép theo cache ở lượt đi.** Ép $a^{(l,+)} - a^{(l,-)} = a^{(l)}$ ở mỗi tầng là một gauge hằng (§2.2) có tham số được chọn nhờ đáp án của mạng gốc. Nó giữ $\chi$ nhỏ *vì đã biết trước* $a^{(l)}$, và tốn bộ nhớ lưu kích hoạt cùng gradient của mọi tầng.

**Dịch $\alpha$ ở lượt về.** Công thức dạng đóng [HM, Thm 9]: đặt $\bar\alpha_j := \prod_{j\le i\le L}\big(1-2\alpha^{(i)}\big)$, và gọi $Q[j]$ là mạng gốc trong đó, từ tầng $j$ trở đi, ReLU được thay bằng $\min\{x,0\}$ và trọng số các tầng $j+1,\dots,L$ được thay bằng trị tuyệt đối. Khi đó
$$\delta^g_{\text{shift}} = \tfrac12\Big[\delta - \sum_{0\le j<L}\bar\alpha_j\,\frac{\partial Q[j+1]}{\partial x}\Big], \qquad \delta^h_{\text{shift}} = -\tfrac12\Big[\delta + \sum_{0\le j<L}\bar\alpha_j\,\frac{\partial Q[j+1]}{\partial x}\Big], \qquad \delta = \nabla f.$$
Các số hạng $\partial Q[j]/\partial x$ tăng nhanh khi $j \to 0$ [HM, Rem. 2].

- $\alpha = 0$: $\bar\alpha_j = 1$, đạo hàm nổ. Với SplitGrad (26,g), Pointing Game giảm từ $0.956$ ($\alpha = 0.35$) xuống $0.401$ ($\alpha = 0$) [HM, Tab. 3].
- $\alpha = 0.5$: $\bar\alpha_j = 0$, độ nhạy suy biến thành $\pm\tfrac12\nabla f$, tức chỉ còn là bội của gradient gốc.

> **Quan sát (tự suy ra từ [HM, Thm 9] — cần kiểm lại): $\alpha$-shifting là một phép chọn gauge ở mức đạo hàm.**
>
> Với mọi $\alpha$, $\delta^g_{\text{shift}} - \delta^h_{\text{shift}} = \delta = \nabla f$. So với cặp gradient không dịch $(\nabla g, \nabla h)$ (trường hợp $\alpha = 0$),
> $$\big(\delta^g_{\text{shift}},\ \delta^h_{\text{shift}}\big) = \big(\nabla g,\ \nabla h\big) + v_\alpha\,(1,1), \qquad v_\alpha = \tfrac12\sum_{0\le j<L}\big(1-\bar\alpha_j\big)\frac{\partial Q[j+1]}{\partial x}.$$
> Như vậy $\alpha$ chỉ cộng **cùng một véc-tơ** $v_\alpha$ vào cả hai luồng. Tại $x^\star$, điều này tương đương với cộng gauge affine $C(x) = \langle v_\alpha, x\rangle$ (lồi, $D_C \equiv 0$) — cặp mới vẫn lồi nhưng có thể mất tính đơn điệu. Hệ quả:
>
> 1. Không tồn tại "đạo hàm thực" của $g$ hay $h$ độc lập với gauge. Lập luận "α làm lệch đạo hàm thực của thành phần lồi/đơn điệu" cần viết lại thành "α **chọn gauge một cách thực nghiệm**, không có nguyên lý".
> 2. Mọi thông tin mà SplitGrad bổ sung so với $\nabla f$ nằm trọn trong phần chung $\delta^g + \delta^h$, tức phần gauge.

---

## 4. Phương pháp đề xuất: Phân rã DC dịch tâm gauge (Gauge-Centred DC Decomposition) cho XAI

*Nguồn: [P1, §2.4–2.6], [DCD, §5], kết nối với các công cụ XAI của [HM].*

### 4.1 Nguyên lý dịch tâm gauge

**Nguyên nhân gốc** [P1, Rem. 14]. Theo (8), thành phần âm *chính là* gauge, và điều kiện hoá *chính là* gauge đo so với hàm. Quy tắc nhân (3) dựng gauge từ các bình phương đặt tâm tại **gốc** không gian, nên tại điểm cần giải thích $x^\star$ cả $g(x^\star)$ và $h(x^\star)$ đều khổng lồ.

**Mệnh đề 17 (dạng tổng quát, không cần dựng đệ quy) [P1].** Nếu $f \in C^2$ trên quả cầu $\bar\Omega = \{\|x - x^\star\| \le \rho\}$ và $C \ge \sup_{\Omega}\|\nabla^2 f(x)\|_{\mathrm{op}}$, thì
$$h_{x^\star}(x) = \frac C2\|x - x^\star\|^2, \qquad g_{x^\star}(x) = f(x) + \frac C2\|x-x^\star\|^2$$
là một cặp DC hợp lệ của $f$ trên $\Omega$ (vì $\nabla^2 g_{x^\star} = \nabla^2 f + CI \succeq 0$), với
$$\chi_{\mathrm{DC}}(x^\star) = 1, \qquad \sup_\Omega \chi_{\mathrm{DC}} \le 1 + \frac{C\rho^2}{\inf_\Omega\lvert f\rvert}.$$

**Định lý 12 (chuỗi — điều kiện hoá tối ưu) [P1].** Trên $\Omega_\rho = (1-\rho,\,1+\rho)^L$ với $f_L(w) = \prod_j w_j$, đặt $C_{L,\rho} = (L-1)(1+\rho)^{L-2}$ (cận Gershgorin), $Q^\rho(w) = \frac{C_{L,\rho}}{2}\|w - \mathbf 1\|^2$ và $P^\rho = f_L + Q^\rho$. Với $k = \lfloor L/2\rfloor$ và $\gamma = \frac{1+\rho}{1-\rho}$:

1. $(P^\rho, Q^\rho)$ là cặp NDC của $f_L$ trên $\Omega_\rho$;
2. $\chi^{\mathrm{pt}}_L(\rho) = 1$, giá trị tốt nhất có thể, đạt tại tâm;
3. $2e^{k\rho^2} - 1 \ \le\ \chi^{\mathrm{unif}}_L(\rho) \ \le\ 1 + \dfrac{L(L-1)\rho^2(1+\rho)^{L-2}}{(1-\rho)^L} \ \le\ 2L^2\gamma^L$;
4. $Q^\rho$ có bậc 2 và $P^\rho$ có bậc $L$, đúng bằng bậc của $f_L$.

**Hệ quả 13 và 15.** Độ sâu chứng nhận được với $b$ bit tăng từ $\Theta(\log b)$ (quy tắc naive) lên $\Theta(b)$ (gauge-centred); điều kiện đủ là $b \ge L\log_2\gamma + 2\log_2 L + 1 + \log_2(1/\delta)$. Kết quả này tối ưu theo bậc: trên một khối lập phương bán kính cố định, *không gauge nào* xoá được mũ đơn còn lại, vì $\sup\chi_{\mathrm{DC}} \ge 2e^{\lfloor L/2\rfloor\rho^2} - 1$. Gauge centring xoá đúng mũ thứ hai.

**Định lý 18 (DCA gauge-centred = proximal-point) [P1].** Giả sử $f \in C^2$ và $\nabla^2 f \succeq -\mu I$ trên $\Omega$ (ký hiệu $\mu$ để tránh trùng với bán kính $\rho$), với $C \ge \mu$. Khi đó:

1. Bước DCA $x_{k+1} \in \arg\min_z\{P(z) - \langle\nabla Q(x_k), z\rangle\}$ chính là bước proximal-point
   $$x_{k+1} \in \arg\min_{z\in\Omega}\Big\{f(z) + \frac C2\|z - x_k\|^2\Big\},$$
   và không phụ thuộc vào tâm $x^\star$: *tâm quyết định điều kiện số, còn $C$ quyết định trọng số proximal*.
2. Bất biến điểm tới hạn: với mọi phân rã gốc $g_0 - h_0$ và mọi gauge lồi $c \in C^1$, $\operatorname{Crit}\big((g_0+c)-(h_0+c)\big) = \operatorname{Crit}(g_0 - h_0)$.
3. Bài toán con $(C-\mu)$-lồi mạnh và giảm đơn điệu: $f(x_k) - f(x_{k+1}) \ge \frac C2\|x_{k+1}-x_k\|^2$.
4. Hội tụ: $\sum_k\|x_{k+1}-x_k\|^2 < \infty$ và mọi điểm tụ là điểm tới hạn; nếu thêm tính chất Kurdyka–Łojasiewicz thì toàn dãy hội tụ.

Ý nghĩa với XAI: tìm giải thích phản thực tế (counterfactual) hoặc mẫu đối kháng bằng DCA trên cặp $(g_{x^\star}, h_{x^\star})$ vẫn chạy được ở những độ sâu mà DCA naive đã tràn số. Từ $L \ge 12$, hệ số tuyến tính hoá naive đã là $\pm\infty$/NaN [P1, Tab. 7].

**Thủ tục 16 (chứng nhận DC gauge-centred) [P1].** Đầu vào: $f$, $x^\star$, bán kính $\rho$, độ chính xác mục tiêu $\delta$ và đơn vị làm tròn $u$.

1. Tính một cận độ cong tường minh $C \ge \sup_{\Omega_\rho}\|\nabla^2 f\|_{\mathrm{op}}$ (Gershgorin, interval arithmetic, hoặc cận Lipschitz của $\nabla f$). Đây là một phép tính cận một lần, không phải bài toán tối ưu.
2. Đặt $Q(x) = \frac C2\|x - x^\star\|^2$ và $P = f + Q$.
3. Chứng nhận: sai số tương đối $\le u\,\chi_{\mathrm{DC}}$, trong đó $\chi_{\mathrm{DC}}(x^\star) = 1$.

Ước lượng thừa $C$ vẫn an toàn — cái giá chỉ là $\chi$ trên vùng lớn hơn và DCA cần nhiều bước hơn. Chọn $C < C_{L,\rho}$ (làm $P$ mất tính lồi) là trường hợp hỏng duy nhất [P1, Tab. 9].

**Bảo đảm tại điểm và trên vùng** [P1, §2.4]. $\chi_{\mathrm{DC}}(x^\star) = 1$ chỉ khẳng định *phép trừ* $P - Q$ tại $x^\star$ không bị triệt tiêu. Nó không nói rằng Hessian hay bài toán tối ưu có điều kiện tốt. Ra ngoài $\Omega_\rho$, các ReLU có thể đổi trạng thái, khi đó phải đặt lại tâm.

**Mở rộng cho softmax attention** [P1, §2.4, §3.7]. Hessian của log-sum-exp là $\frac1\tau(\operatorname{diag}p - pp^\top) \preceq \frac1{2\tau}I$, không phụ thuộc độ dài chuỗi, nên khối attention có cận $C$ dạng đóng. Softmax không phải nút tự củng cố — chỉ $QK^\top$ và $AV$ là. Vì vậy không cần dựng lift DC đệ quy cho softmax. Thực nghiệm trên stack softmax attention thật cho sai số $\sim 10^{-16}$ cả tại tâm lẫn tại góc vùng, với $N = 1..6$ khối [P1, Tab. 12].

**Tính cận độ cong cho kích hoạt trơn** [DCD, Thm 5.10, Cor. 5.8–5.9]. Với $\Psi = \Phi_\sigma(P,Q)$, $D = P - Q$ và $\alpha = \sigma'(D)$:
$$\nabla^2\Psi = \alpha\nabla^2 P + (1-\alpha)\nabla^2 Q + \sigma''(D)\,(\nabla P - \nabla Q)(\nabla P - \nabla Q)^\top.$$
Softplus: $\sup_t\lvert\sigma_\tau(t) - t_+\rvert = \tau\log 2$ và $\sup\sigma_\tau'' = \frac1{4\tau}$. Làm trơn Gauss: $\frac{\varepsilon}{\sqrt{2\pi}}$ và $\frac{1}{\sqrt{2\pi}\,\varepsilon}$. Đây là sự đánh đổi giữa sai số xấp xỉ và độ cong.

**Tóm tắt các tính chất của $(g_{x^\star}, h_{x^\star})$**

1. **Lồi và chính xác:** cả hai lồi trên $\Omega_\rho$ và $g_{x^\star} - h_{x^\star} = f$ chính xác tuyệt đối.
2. **$\chi_{\mathrm{DC}}(x^\star) = 1$:** $h_{x^\star}(x^\star) = 0$ và $\nabla h_{x^\star}(x^\star) = 0$, nên $g_{x^\star}(x^\star) = f(x^\star)$ và $\nabla g_{x^\star}(x^\star) = \nabla f(x^\star)$ ở mọi độ sâu $L$.
3. **Loại một tầng hàm mũ trên $\Omega_\rho$:** từ mũ kép xuống mũ đơn; độ sâu khả thi tăng từ $\Theta(\log b)$ lên $\Theta(b)$, tối ưu theo bậc (Hệ quả 13, 15).
4. **Bảo toàn cấu trúc điểm dừng:** DCA trên cặp này là proximal-point với bài toán con lồi mạnh (Định lý 18).

### 4.2 Tích hợp gauge centring vào các phương pháp giải thích (Gauge-Grad, Gauge-CAM, Gauge-LRP)

**Cấp toàn cục / khối (non-recursive, block-wise — Mệnh đề 17).** Thay vì tách đệ quy qua từng phép nhân trong khối gated hay softmax attention (nguồn gốc của mũ kép), áp gauge centring trực tiếp cho từng khối hoặc tại tầng trích xuất đặc trưng $\ell$, với cận độ cong giải tích (cận Gershgorin; cận $\frac1{2\tau}I$ cho softmax).

**Cấp luồng tách từng tầng (layer-wise gauge-centred split-streams).** Khi lấy đặc trưng đơn điệu tại tầng trung gian $l$ (ví dụ tầng 14 hoặc 26 của VGG16) từ cặp luồng $(a^{(l,+)}, a^{(l,-)})$ của [HM], thay phép co giãn ngưỡng $(\theta, \Theta)$ và phép dịch $\alpha$ bằng dịch tâm gauge tại kích hoạt $a^{(l,\star)}$, sao cho thành phần dư triệt tiêu đúng tại tâm.

**Các công thức gốc của [HM] làm khuôn:**

- LayerCAM: $\displaystyle \mathrm{LayerCAM}^{(l)} = \operatorname{ReLU}\Big(\sum_c \frac{\partial y}{\partial a_c^{(l)}} \odot a_c^{(l)}\Big)$.
- SplitCAM (bỏ ReLU): $\displaystyle \mathrm{SplitCAM}^{(l,+,g)} = \sum_c \delta^{(l,+,g)}_{\text{shift}} \odot a_c^{(l,+)}$ và $\displaystyle \mathrm{SplitCAM}^{(l,g)} = \sum_c \delta^{(l,g)}_{\text{shift}} \odot a_c^{(l)}$.
- SplitGrad: trung bình theo kênh của gradient split-pair đã dịch.
- LRP-$\gamma$: $\tilde z_{ij} = a_j\big(W_{ij} + \gamma\max\{W_{ij},0\}\big)$ và
  $$R_j^{(l)} = \sum_i \frac{\tilde z_{ij}}{\sum_{j'}\tilde z_{ij'} + \varepsilon\operatorname{sign}\big(\sum_{j'}\tilde z_{ij'}\big)}\,R_i^{(l+1)}.$$
  SplitLRP dùng $\gamma = 0$, $\varepsilon = 10^{-6}$ trên split-pair, xuất phát từ $a^{(L,+)}$, với bản đồ kết hợp $R^{(l,\text{comb})} = R^{(l,+)} - R^{(l,-)}$.

**Ba bộ giải thích đối ứng trực tiếp với [HM]:**

- **Gauge-SplitGrad:** trích độ nhạy từ cặp đã dịch tâm gauge, không cần hệ số $\alpha$.
- **Gauge-SplitCAM:** $\sum_c \delta^{(l,g)}_{\text{gauge}} \odot a_c^{(l,\star)}$ theo kiểu LayerCAM.
- **Gauge-SplitLRP:** lan truyền relevance trên cặp luồng đã dịch tâm. Mục tiêu là khắc phục bản đồ trắng vô nghĩa (khi thiếu ổn định lượt đi) và sự trơ với ngẫu nhiên hoá trọng số các tầng cuối (model randomization) của SplitLRP gốc.

Lưu ý: $\delta_{\text{gauge}}$ chưa được định nghĩa — xem §4.3.

### 4.3 ⚠ Vấn đề mở — cần thảo luận trước khi viết

1. **Suy biến tại $x^\star$.** Vì $\nabla h_{x^\star}(x^\star) = 0$ và $\nabla g_{x^\star}(x^\star) = \nabla f(x^\star)$, nếu Gauge-SplitGrad đọc gradient của cặp dịch tâm tại $x^\star$ thì nó **trùng với Saliency** ($\nabla f$): luồng $h$ (counterfactual) biến mất. Đây đúng là sự suy biến mà §3.3 phê phán ở $\alpha = 0.5$. Theo Quan sát ở §3.3, hai trường hợp chỉ khác nhau một gauge affine, và cả hai đều không mang thông tin nào ngoài $\nabla f$. Ngoài ra $g_{x^\star}$ không còn đơn điệu ($\nabla f$ có thể âm), nên mất tính chất "hidden monotonicity" [HM, Thm 2]. Cần một gauge vừa có điều kiện tốt vừa giữ được thông tin tách — hiện chưa có đề xuất.
2. **ReLU không thuộc $C^2$.** Mệnh đề 17 đòi hỏi $f \in C^2$, trong khi VGG16 và ResNet18 là hàm CPWL:
   - trong vùng tuyến tính chứa $x^\star$, $f$ là affine nên $C = 0$ và $h \equiv 0$ (tầm thường);
   - trên vùng có chứa một nếp gấp lõm, **không** $C$ hữu hạn nào lồi hoá được $f + \frac C2\|x - x^\star\|^2$ (ví dụ $f = -\lvert x\rvert$).

   Các lựa chọn khả dĩ:
   - (a) thay ReLU bằng softplus hoặc ReLU làm trơn Gauss ($\sigma'' \le \frac1{4\tau}$, sai lệch $\le \tau\log 2$ mỗi đơn vị), rồi lập luận rằng mạng xấp xỉ vẫn giải thích được mạng gốc;
   - (b) dùng gauge CPWL lồi thay cho gauge bậc hai;
   - (c) chỉ áp gauge centring cho các khối có nút nhân (gated, attention) và giữ cách tách của [HM] cho phần CPWL.
3. **VGG16/ResNet18 thuộc chế độ mũ đơn.** [HM] đã xử lý được chế độ này bằng float64 + cache. Bản thân giá trị $f(x^\star)$ có sẵn từ một lượt forward — khôi phục nó qua $P - Q$ chỉ là "nhiệt kế, không phải mục tiêu" [P1, §3.9]. Lợi thế số học rõ rệt của gauge centring nằm ở các kiến trúc có nút nhân (mạng cổng, ViT), nên trọng tâm bài báo nên đặt ở đó.
4. **Cận trên vùng** cần $\inf_{\Omega_\rho}\lvert f\rvert > 0$. Logit gần $0$ làm cận $\sup\chi_{\mathrm{DC}}$ trở nên vô dụng.
5. **So sánh tốc độ.** [HM, B.5] tự nhận cài đặt của họ chưa tối ưu (nhân đôi ma trận trọng số, hook thủ công). Muốn so sánh tốc độ công bằng thì phải cài đặt lại [HM] một cách tối ưu.

---

## 5. Thực nghiệm

Phần thực nghiệm chia hai nhóm: **(A)** chẩn đoán số học, kiểm chứng lý thuyết trên chính cơ chế của [HM]; **(B)** benchmark XAI chuẩn, bám sát thiết lập của [HM].

### Nhóm A — Chẩn đoán số học: (A) Naive DC vs. (B) HM stabilization vs. (C) Gauge centring

*Kế thừa [P1, §3] và [HM, App. C.1–C.2].*

**Đại lượng đo.** Sai số tương đối của giá trị và gradient so với ground truth độ chính xác cao (mpmath 40–60 chữ số, như [P1]); $\chi_{\mathrm{DC}}$ theo độ sâu; định dạng float64/float32/float16.

**Mô hình**

- Chuỗi ReLU sâu (6).
- Mạng cổng [P1, (11)]: $h_0(x) = 1$, $h_\ell(x) = \operatorname{ReLU}\big(g_\ell(x)\,h_{\ell-1}(x)\big)$, $g_\ell(x) = \operatorname{ReLU}(w_\ell x + b_\ell)$, $f = h_L$.
- Các tầng sâu của VGG16 và ResNet18.
- Khối multi-head attention [P1, §3.6], cả linear và softmax:
  $$S_h = \tfrac{1}{\sqrt{d_h}}Q_hK_h^\top,\quad O_h = S_hV_h,\quad X \leftarrow X + [O_1\cdots O_H]W_O,\quad X \leftarrow X + \operatorname{ReLU}(XW_1)W_2;$$
  sau đó là ViT.

**Đối thủ**

- **(A) Naive split/DC:** phân tách đệ quy, không ổn định hoá.
- **(B) HM stabilization:** co giãn lượt đi với $\theta = 0.1$, $\Theta = 10$; dịch lượt về với $\alpha \in \{0, 0.3, 0.4, 0.5\}$. **Không** dùng cache mạng gốc để ghi đè, nhằm làm lộ rõ bản chất số học.
- **(C) Gauge centring (ours).**

**Mốc tham chiếu từ [P1]** — mạng cổng, float64, sai số tương đối so với ground truth 60 chữ số [P1, Tab. 5]:

| $L$ | $\chi_{\mathrm{DC}}$ | $f(x^\star)$: A | B | C | $f'(x^\star)$: A | B | C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | $1.3\times10^6$ | 8.8e−12 | 8.8e−12 | 2.9e−17 | 3.7e−10 | 3.7e−10 | 1.5e−16 |
| 6 | $4.6\times10^{12}$ | 6.6e−4 | 6.2e−5 | 4.0e−17 | 1.1e−3 | 3.8e−4 | 2.4e−16 |
| 7 | overflow | 1.0 | 1.0 | 7.7e−17 | 1.0 | 1.0 | 2.3e−16 |
| 40 | overflow | 1.0 | 1.0 | 7.6e−17 | 1.0 | 1.0 | 8.9e−17 |

Độ sâu chứng nhận tối đa trên $\Omega_{0.05}$ với sai số $< 10^{-2}$ [P1, Tab. 8]:

| Định dạng | Naive | Gauge-centred |
| --- | --- | --- |
| float16 | 3 | 100 |
| float32 | 5 | > 119 |
| float64 | 6 | > 119 |

**Ablation gauge** [P1, Tab. 9], cần lặp lại: thay đổi hệ số $C = m\,C^\star$; vị trí tâm (gốc cho $\chi \approx 4.5\times10^3$ tại $L = 24$, so với $1.0$ khi đặt tâm tại $x^\star$); bán kính $\rho$.

**Kết quả kỳ vọng.** Trên mạng cổng và ViT ($L \gtrsim 4$–$7$), (A) và (B) sụp về sai số tương đối $1.0$ — mất 100% chữ số có nghĩa của đạo hàm — hoặc (B) chỉ giữ được biên độ nhờ thay đổi gauge theo $\alpha$. Trong khi đó (C) giữ sai số $\sim 10^{-16}$ ở mọi độ sâu. Trên VGG16 (tầng 14–26, chế độ mũ đơn), kỳ vọng (A) ≈ (B) mất nhiều chữ số khi không có cache; lưu ý vấn đề 2 ở §4.3 — (C) cần phiên bản làm trơn.

### Nhóm B — Đánh giá giải thích mô hình (XAI benchmark như [HM])

*Bám sát [HM, §5 và App. B, C, D].*

**Mô hình và dữ liệu**

- **CNN chuẩn:** VGG16 và ResNet18 pretrained trên ImageNet, đánh giá trên ImageNet-S-50 (mặt nạ phân vùng ground-truth của 50 lớp): 50 ảnh validation (một ảnh mỗi lớp, dùng để chọn tầng/cấu hình) và 566 ảnh test — giống hệt thiết lập của [HM].
- **Mở rộng (điểm nhấn):** một mô hình có nút nhân tự củng cố — ViT, hoặc Multiplicative Maxout net [HM, App. E] — để cho thấy gauge centring mở khoá khả năng giải thích cho họ kiến trúc có nút nhân. Lưu ý: [HM] trích Mao et al. (2022) làm nguồn của ImageNet-S-50 chứ **không** thử ViT; việc áp dụng cho Transformer được [HM, §6] để ngỏ.
- **Factual vs. counterfactual trên MNIST** [HM, App. D]:
  - kiến trúc: `cnn_maxout` (4 tầng conv 32/64/128/128, FC $6272\to1024\to256\to10$, rank-2 Maxout) và `mlp_maxout` ($784\to800\to800\to800\to10$);
  - các mô hình DIC/DM và biến thể đảo đầu vào DICIN/DMIN (luồng $h$ nhận $1-x$);
  - đối chiếu trực quan vùng nét chữ hiện hữu ($g$-stream) với vùng nét chữ bị khuyết ($h$-stream).

**Các phương pháp so sánh**

- **Classical XAI:** Saliency, Input × Gradient, Integrated Gradients, Gradient SHAP, DeepLift, GradCAM, GradCAM++, LayerCAM, Guided Backprop, Guided GradCAM, Deconvolution, LRP ($\gamma = 0.25$, Zennit), Occlusion, Feature Ablation (Captum).
- **Hidden Monotonicity:** SplitCAM, SplitGrad, SplitLRP với $\alpha \in \{0.3, 0.35, 0.4, 0.5\}$ và các biến thể sc/wta/abs:
  - VGG16: SplitCAM/SplitGrad tại các tầng (12,g), (14,g), (26,+,g); SplitLRP tại (7,pos), (19,pos), (26,pos);
  - ResNet18: các khối (1.0.2,g) và (4.1.2,g).
- **Đề xuất (ours):** Gauge-CAM, Gauge-Grad, Gauge-LRP.

**Chọn tầng tự động** [HM, §5.2]. Với mỗi phương pháp, chọn cấu hình tầng/abs có (i) hạng trung bình tốt nhất trên hai chỉ số định vị và (ii) hạng trung bình tốt nhất trên Pixel Flipping; tinh chỉnh trên tập validation, báo cáo trên tập test.

**Hệ thống thước đo Quantus** — đúng 6 thước đo thuộc 3 nhóm mà [HM] dùng:

| Nhóm | Thước đo | Hướng | Ghi chú |
| --- | --- | --- | --- |
| Định vị | Attribution Localization | ↑ | tỉ lệ khối lượng attribution nằm trong mặt nạ |
| Định vị | Pointing Game | ↑ | pixel có attribution lớn nhất có nằm trong mặt nạ không |
| Trung thực | Selectivity | ↓ | |
| Trung thực | Pixel Flipping AUC@5, AUC@20 | ↑ | 112 pixel mỗi bước, thay bằng giá trị trung bình ảnh |
| Bền vững | Maximum Sensitivity | ↓ | 25 mẫu nhiễu mỗi ảnh |

**Mốc cần vượt** — tốt nhất của Split methods so với tốt nhất của baseline [HM, Tab. 1–2]:

| | Select. ↓ | Attr. Loc. ↑ | Point. ↑ | PF@5 ↑ | PF@20 ↑ | Max Sens. ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| VGG16 — Split | 2.727 | 0.651 | 0.938 | 0.300 | 2.404 | 0.282 |
| VGG16 — baseline | 2.792 | 0.632 | 0.887 | 0.317 | 2.474 | 0.375 |
| ResNet18 — Split | 3.503 | 0.673 | 0.972 | 0.168 | 1.395 | 0.219 |
| ResNet18 — baseline | 3.517 | 0.664 | 0.971 | 0.213 | 1.757 | 0.131 |

Trên ResNet18, Split methods chỉ nhỉnh hơn baseline ở Selectivity và Pointing Game; Pixel Flipping và Max Sensitivity thua baseline.

**Sanity check và ablation**

- **Randomization sanity check** [HM, App. B.4, Tab. 6]. Khởi tạo lại ngẫu nhiên (Gaussian) trọng số 3 tầng FC cuối của VGG16; đo HOG, Spearman (có dấu) và SSIM giữa bản đồ trước và sau ngẫu nhiên hoá — độ tương đồng thấp hơn là tốt hơn.
  - SplitLRP thất bại: HOG ≈ 0.99, Spearman ≈ 1.00, SSIM ≈ 0.45–0.56. [HM] quy nguyên nhân cho hiệu ứng *diminishing gradient* do ma trận trọng số không âm (ε-LRP trên split-stream gần với Input × Gradient).
  - SplitCAM/SplitGrad: Spearman 0.096–0.729, SSIM 0.091–0.306.
  - Giả thuyết cần kiểm: gauge centring khắc phục điểm mù này. Lưu ý: nếu phương pháp suy biến về Saliency (§4.3, vấn đề 1) thì sẽ qua sanity check một cách tầm thường — bản thân Saliency có HOG 0.188, Spearman 0.416, SSIM 0.353.
- **Loại bỏ phụ thuộc vào $\alpha$** [HM, Tab. 3, Tab. 8].
  - SplitGrad (26,g): Pointing Game theo $\alpha = 0.5 \to 0$ là $0.898,\ 0.906,\ 0.947,\ \mathbf{0.956},\ 0.889,\ 0.401$; Max Sensitivity lại tốt nhất ở $\alpha = 0$ ($0.028$).
  - SplitCAM (26,+,g): Pointing Game $0.868 \to 0.938$ ($\alpha = 0.4$) $\to 0.613$ ($\alpha = 0.3$) $\to 0.855$ ($\alpha = 0$) — không đơn điệu theo $\alpha$.
  - Mục tiêu: phương pháp không cần tinh chỉnh $\alpha$ và ổn định qua mọi tầng, từ nông đến sâu.
- **Hiệu năng tính toán** [HM, App. B.5, Tab. 7], đơn vị giây/ảnh:

  | Phương pháp | ResNet18 | VGG16 |
  |---|---|---|
  | Model splitting (một lần) | 0.0331 | 0.0295 |
  | SplitCAM | 0.4368 | 0.3348 |
  | SplitGrad | 0.4284 | 0.3309 |
  | SplitLRP | 0.5711 | 0.3204 |
  | LayerCAM | 0.0149 | 0.0124 |
  | LRP $\gamma = 0.25$ | 0.0599 | 0.0507 |
  | Saliency | 0.0060 | 0.0047 |

  SplitCAM chậm hơn LayerCAM khoảng $27$–$29\times$; tham số $\alpha$ không ảnh hưởng thời gian chạy. [HM] quy chi phí cho cách cài đặt chưa tối ưu (nhân đôi trọng số, forward/backward thủ công) chứ không riêng gì float64. Gauge centring ở cấp khối (Thủ tục 16 / Mệnh đề 17) không cần nhân đôi đồ thị và có thể chạy trên float32 — nhưng điều này phải được đo, không chỉ suy ra.
