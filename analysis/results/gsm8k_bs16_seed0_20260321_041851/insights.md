# Insights

## Pham vi
Ket qua trong thu muc nay duoc sinh boi `Analysis/visualize.py`. Day khong phai la phep so sanh truc tiep voi Dflash, ma la phep mo xet rieng co che tron VCD thong qua 3 nhanh:

- `positive`: draft logits goc.
- `negative`: logits tu negative context.
- `final`: logits sau khi tron contrastive.

## Ket qua chinh
- Co `6074` step record, trong do `5000` step duoc dua vao bao cao.
- Mean acceptance length:
  - positive: `5.3062`
  - negative: `0.8314`
  - final: `5.2858`
- `78.70%` step co `positive_accept_len > negative_accept_len`.
- `97.32%` step co `final_accept_len <= positive_accept_len`.
- Mean KL giua positive va negative la `2.8059`, max `18.25`.
- Tinh tren toan bo `6074` step record, gain `final - positive` chi la `-0.0207`.
- `92.79%` step giu nguyen acceptance, chi `2.75%` step cai thien, trong khi `4.46%` step bi xau di.

## Dien giai
Negative branch khong phai vo nghia: KL trung binh ~`2.81` cho thay positive va negative khac nhau dang ke, va acceptance cua negative cung rat thap (`0.83`). Van de nam o cho su khac biet nay khong duoc chuyen thanh loi ich khi tron vao final branch.

Noi cach khac, contrastive signal dang tao ra "divergence" nhung khong tao ra "better accepted prefix". Final branch gan nhu trung voi positive branch, va o nhung buoc thuc su khac thi xac suat lam xau cao hon xac suat lam tot.

## Tai sao dieu nay quan trong cho bai toan "vi sao chua hon Dflash"
Neu rieng phep tron VCD con khong vuot duoc positive branch trong cung mot decode loop, thi rat kho ky vong no vuot duoc Dflash o muc case-level. File nay cho thay van de cot loi khong nam o cho negative branch qua giong positive, ma la o cho co che tron contrastive hien tai chua bien su khac biet thanh prefix hop le hon de target verifier chap nhan.

## Gia thuyet co so du lieu ho tro
- O nhung step KL rat cao (quartile cao nhat, `3.7344 -> 18.25`), mean gain giam manh nhat: `-0.0494`.
- Nghia la khi contrastive tac dong manh hon, no thuong gay lech nhieu hon la sua dung.
- Final branch vi vay dang co xu huong "an mon" positive branch, khong phai "bo sung" cho no.

## Ket luan ngan
Thu muc nay cung cap bang chung rang van de cua VCD hien tai la chat luong phep tron. Contrastive branch co tin hieu, nhung tin hieu do chua du on dinh de tang acceptance; da so truong hop final bang positive, va khi khac thi thuong bat loi hon la co loi.
