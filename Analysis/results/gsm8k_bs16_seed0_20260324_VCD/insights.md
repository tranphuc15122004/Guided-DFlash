# Insights

## Pham vi
Ket qua trong thu muc nay duoc sinh boi `Analysis/VCD_vs_Dflash.py`, so sanh truc tiep `VCD + Dflash` voi `Dflash` tren `128` case cua `gsm8k`.

## Ket qua chinh
- Mean acceptance cua Dflash: `6.4979`
- Mean acceptance cua VCD+Dflash: `6.4783`
- Mean delta `VCD - Dflash`: `-0.0196`
- So case:
  - VCD tot hon: `42`
  - Dflash tot hon: `48`
  - Hoa: `38`
- Median delta bang `0.0`
- Khi VCD thang, muc tang trung binh la `+0.2008`; khi thua, muc giam trung binh la `-0.2281`.
- `case_records.jsonl` cho thay `output_text` cua VCD va Dflash giong nhau `128/128` case.

## Theo pattern
- `math|<=128`: delta `-0.0177` tren `120` case
- `math|129-512`: delta `-0.0556` tren `6` case
- `coding|<=128`: delta `-0.0276` tren `2` case

VCD la bien the "it hai hon CD" trong tap nay, nhung van chua vuot Dflash tren trung binh o bat ky bucket nao.

## So sanh voi CD
Khi doi chieu 2 file `case_records` cua CD va VCD tren cung `128` case:

- VCD tot hon CD o `39` case
- CD tot hon VCD o `12` case
- Hai ben bang nhau o `77` case
- Mean `(VCD - CD)` ve delta acceptance la `+0.0611`

Nghia la VCD da giai quyet duoc mot phan tac dung phu cua CD, nhung phan cai thien them nay van chua du de vuot qua baseline Dflash.

## Dien giai
VCD khong that bai theo kieu "qua pha". Nguoc lai, no rat gan Dflash:

- delta trung binh chi `-0.0196`
- so case thang/thua kha can bang (`42` vs `48`)
- nhieu case hoa (`38`)

Dieu nay cho thay co che VCD co huong dung hon CD, nhung van gap mot tran tran: no chua tao ra du buoc prefix hop le hon de target verifier chap nhan nhieu hon Dflash.

## Tai sao VCD van chua hon Dflash
1. Gain co, nhung qua nho.
Muc cai thien o cac case thang co that, nhung khong du bu muc mat o cac case thua.

2. Divergence manh khong dong nghia voi acceptance tot hon.
Correlation giua `avg_kl_divergence` va delta la `-0.0470`. Quartile KL cao nhat co mean delta xau nhat: `-0.0556`.

3. VCD van chua thay doi duoc quy dao sinh cuoi cung.
Output text trung Dflash `100%`, nen VCD hien tai chi tac dong vao acceptance dynamics. Khi dynamics nay chi "gan nhu bang" Dflash, tong the se van khong thang.

4. Ket qua trong `gsm8k_bs16_seed0_20260321_041851` giai thich them.
Phan diagnostic cho thay final VCD branch hau nhu khong vuot positive branch ngay trong mot decode step. Vi vay o muc case-level, VCD chi co the tiem can Dflash, khong de dang vuot qua.

## Ket luan ngan
VCD hien tai la cach ket hop contrastive tot nhat trong nhung file ket qua nay, nhung van chua hon Dflash. Ly do khong phai vi VCD hoan toan vo tac dung, ma vi tac dung cua no qua yeu va qua khong on dinh de chuyen thanh loi ich acceptance tong the.
