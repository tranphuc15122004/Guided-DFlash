# Insights

## Pham vi
Ket qua trong thu muc nay duoc sinh boi `Analysis/CD_vs_Dflash.py`, so sanh truc tiep `CD + Dflash` voi `Dflash` tren `128` case cua `gsm8k`.

## Ket qua chinh
- Mean acceptance cua Dflash: `6.4979`
- Mean acceptance cua CD+Dflash: `6.4172`
- Mean delta `CD - Dflash`: `-0.0807`
- So case:
  - CD tot hon: `31`
  - Dflash tot hon: `62`
  - Hoa: `35`
- Median delta bang `0.0`, nghia la phan lon case hoac hoa, hoac chenh lech rat nho.
- Khi CD thang, muc tang trung binh chi `+0.1955`; khi CD thua, muc giam trung binh la `-0.2645`.
- `case_records.jsonl` cho thay `output_text` cua CD va Dflash giong nhau `128/128` case.

## Theo pattern
- `math|<=128`: delta `-0.0830` tren `120` case
- `math|129-512`: delta `-0.0253` tren `6` case
- `coding|<=128`: delta `-0.1117` tren `2` case

Khong co pattern nao thuc su vuot Dflash tren trung binh. CD giam nhe o moi bucket, va giam ro nhat o bucket toan ngan, la bucket chiem gan nhu toan bo tap danh gia.

## Dien giai
Ket qua nay noi rang CD hien tai khong tao duoc loi ich he thong. No co mot so case thang, nhung:

- So case thua gap doi so case thang (`62` so voi `31`).
- Muc do thua trung binh lon hon muc do thang.
- Output cuoi cung khong doi, nen CD chi dang tac dong vao dong hoc verify/acceptance, khong tao ra huong giai moi tot hon.

Nghia la CD chua giup draft dua ra prefix "de target chap nhan hon". No chi dao dong quanh quyet dinh san co cua Dflash, va tong the thi dao dong nay co xu huong am.

## Tai sao CD chua hon Dflash
Co 3 ly do du lieu ho tro rat ro:

1. Dflash da la baseline manh va on dinh.
Mean acceptance cua Dflash da cao (`6.4979`), nen CD can mot tin hieu rat chinh xac moi vuot duoc. Thuc te, CD khong tao duoc gain du on dinh.

2. Tac dong cua CD khong ben.
Correlation giua `avg_kl_divergence` va delta la `-0.0623`, tuc la divergence lon hon khong dong nghia acceptance tot hon. Quartile KL thu 3 con la bucket xau nhat voi mean delta `-0.1189`.

3. CD khong doi output cuoi cung.
Vi output text trung nhau `100%`, phan khac biet chi nam o prefix acceptance. Neu acceptance da khong tang, thi CD gan nhu khong co co hoi danh bai Dflash o muc he thong.

## Ket luan ngan
CD hien tai la mot phep tron co gia tri chuan doan nhung chua la mot co che giai ma tot hon. No thay doi logit du de gay nhieu case giam acceptance hon tang acceptance, trong khi khong tao ra output cuoi cung moi. Vi vay tren GSM8K, CD hien tai chua the hon Dflash.
