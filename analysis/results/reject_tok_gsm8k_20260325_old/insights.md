# Insights

## Pham vi
Ket qua trong thu muc nay duoc sinh boi `Analysis/check_reject_tok.py`, phan tich token-level cac vi tri bi reject khi dung CD. Luu y run nay dung `cd_alpha = 0.1`, khac voi run case-level trong `gsm8k_bs16_seed0_20260321_CD` (`cd_alpha = 0.5`). Tuy vay, day van la bang chung rat manh cho cac failure mode cau truc cua pipeline CD hien tai.

## Ket qua chinh
- Tong so reject event: `5492`
- Positive hit rate tai vi tri reject: `1.31%`
- CD hit rate: `0.00%`
- Prediction changed rate: `2.55%`
- Target nam trong candidate mask chi `67.64%`
- `cd_overwritten_by_positive_rate`: `68.45%`

Taxonomy reject:

- `overwrite_by_positive_guard`: `3759` event (`68.45%`)
- `no_cd_effect_positive_wrong`: `957` event (`17.43%`)
- `target_filtered_by_candidate_mask`: `659` event (`12.00%`)
- `cd_regression_from_positive`: `72` event (`1.31%`)
- `cd_shifted_but_still_wrong`: `45` event (`0.82%`)

## Ket luan manh nhat
CD hien tai chua thua Dflash chi vi "tin hieu contrastive yeu", ma vi phan lon tac dong cua CD bi khoa hoac bi lam vo hieu ngay trong pipeline:

1. Bi overwrite boi positive logits.
Trong `scheme/CD_inspired.py`, `n_keep = min(6, final_draft_logits.size(1))` va `final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :]`.
Dieu nay dong nghia nhieu vi tri dau trong block bi tra ve positive branch. Phan tich cho thay `68.45%` reject event roi dung vao vung nay.

2. Target token bi candidate mask chan.
Co `12.00%` reject event thuoc taxonomy `target_filtered_by_candidate_mask`, tuc token dung theo posterior bi loai khoi khong gian de cu cua CD truoc khi CD kip sua.

3. O nhieu vi tri con lai, CD gan nhu khong doi top-1.
`17.43%` reject event la `no_cd_effect_positive_wrong`: positive da sai tu dau va CD khong sua duoc.

## Bang chung rang van de la cau truc, khong chi la do alpha
Neu chi nhin vao nhung reject event ma CD thuc su "co co hoi de sua", tuc:

- khong bi overwrite boi positive
- target nam trong candidate mask

thi van chi con `1074 / 5492` event (`19.56%`) la thuc su du dieu kien.

Trong tap "du dieu kien" nay:

- `CD hit = 0`
- `Positive hit = 72`
- CD co doi prediction chi `10.89%`
- target logprob co tang o `71.60%` event, nhung target rank chi cai thien o `3.54%` event

Day la diem rat quan trong: CD thuong tang logprob cua target mot chut, nhung muc tang do khong du de day target len top-1. Nghia la contrastive signal dang "lam mem xac suat", chu chua "doi duoc quyet dinh token".

## Tai sao dieu nay giai thich viec chua hon Dflash
Tu token-level co the rut ra 3 nguyen nhan truc tiep:

1. Qua nhieu vi tri reject nam trong phan ma CD khong duoc phep tac dong.
Neu gan `70%` reject da bi overwrite, CD se khong the tao gain lon o acceptance.

2. Candidate mask hien tai qua chat voi token posterior dung.
Khi target bi loai khoi mask, CD khong con duong nao de chon dung token verifier muon.

3. Ngay ca khi CD duoc phep tac dong, huong sua van chua du manh.
CD co the tang target logprob, nhung hiem khi bien no thanh top-1; khi co doi argmax thi van thuong doi sai huong.

## Noi ket voi ket qua case-level
- O muc case-level, CD thua Dflash trung binh `-0.0807`.
- O muc token-level, CD khong co mot reject nao ma dua duoc target len top-1 (`cd_hit_rate = 0%`).

Hai ket qua nay khop nhau: CD hien tai chua phai mot co che sua draft hieu qua. No bi khoa boi `n_keep`, bi gioi han boi candidate mask, va ngay ca phan con lai cung chua du suc dao chieu quyet dinh verify.

## Ket luan ngan
Thu muc nay la bang chung ro nhat cho cau hoi "tai sao contrastive hien tai chua hon Dflash". Van de chinh khong nam o cho Dflash qua manh mot cach vo ly, ma o cho pipeline CD hien tai tu cat bot phan lon khong gian sua sai, va phan sua sai con lai lai qua yeu de bien thanh token dung.
