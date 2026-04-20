# Bao cao cac giai phap defense prompt injection huu ich cho MedAgentSim

Tai lieu nay co 2 muc tieu:

1. Liet ke cac huong defense dang duoc xem la huu ich, co paper va / hoac repo tham khao ro rang.
2. Danh gia muc do phu hop cua tung huong voi he thong `MedAgentSim`, dac biet trong bai toan:
   - multi-turn medical dialogue
   - attack surface chinh `Patient -> Doctor`
   - co them `Measurement -> Doctor`
   - co `forced final diagnosis`
   - co `internal discussion`

Tai lieu nay uu tien tinh ung dung cho codebase hien co hon la liet ke mang tinh hoc thuat thuần tuy.

---

## 1. Ket luan nhanh

Neu can xep thu tu uu tien cho MedAgentSim, toi khuyen nghi:

1. `Structured separation + provenance marking`
   - hieu qua cao nhat trong nhom black-box / no-training
   - phu hop nhat voi MedAgentSim hien tai
2. `Detector chuyen dung nho + re`
   - dung lam lop bo sung
   - khong nen la defense duy nhat
3. `Output validation / task-level consistency check`
   - rat hop voi bai toan `DIAGNOSIS READY`
4. `Capability / sandbox / least privilege`
   - bat buoc neu sau nay MedAgentSim thanh agent co tool manh hon
5. `White-box retraining`
   - rat manh, nhung chi hop khi ban co ha tang de train / fine-tune

Nghia la: khong co mot detector nao don le du manh. Giai phap thuc dung nhat cho MedAgentSim la `defense-in-depth`.

---

## 2. Nhung defense da co ngay trong codebase nay

### 2.1 `known_answer`

#### Y tuong

Yeu cau model lap lai mot token secret va bo qua phan text ben duoi. Neu model khong lap lai dung token, xem nhu input dang doc hai.

#### Diem manh

- de cai dat
- chi phi tinh toan thap
- co the phat hien mot so payload "lo lieu"

#### Diem yeu

- de bo sot nhieu prompt injection trong multi-turn dialogue
- van dua vao hanh vi tuan lenh cua chinh model dich
- khong xu ly duoc van de tron instruction va data ve mat cau truc

#### Ket qua trong repo

File:

- `MedAgentSim/output_eval_medqa_openpi_reports/report_known_answer.md`

So lieu dang chu y:

- `Avg ASR No Defense = 0.4206`
- `Avg ASR Defense = 0.3112`
- `Avg ASR Reduction = 0.1094`
- `Avg FNR = 0.8126`

#### Danh gia known_answer cho MedAgentSim

- Khong nen dung lam defense chinh.
- Co the giu lai chi de lam baseline so sanh.

---

### 2.2 `llm_based`

#### Y tuong `llm_based`

Dung mot prompt classifier kieu:

- SAFE
- MALICIOUS

roi chan input neu model detector bao malicious.

#### Uu diem `llm_based`

- de hieu
- de benchmark
- voi payload ro rang thi co the bat tot

#### Nhuoc diem `llm_based`

- neu dung cung model / cung backend voi doctor thi hanh vi detector va model dich bi dong bo
- van chi la detector tung turn
- khong tu than giai quyet viec tron lan trusted va untrusted content

#### Danh gia `llm_based` cho MedAgentSim

- Tot hon `known_answer` o mot so thuc nghiem nho.
- Nhung van khong nen dung mot minh.

---

### 2.3 `layered_guard`

#### Y tuong `layered_guard`

Defense moi da duoc them vao codebase nay:

- structured prompt
- provenance marker
- specialized detector / heuristic fallback
- output validation

#### Uu diem `layered_guard`

- phu hop nhat voi codebase hien tai
- khong can fine-tune model
- da phu `measurement`, `forced_final`, `internal_discussion`

#### Nhuoc diem `layered_guard`

- van la black-box defense
- chua co bang so day du full 107-case voi backend song

#### Danh gia `layered_guard` cho MedAgentSim

- Day la defense nen thu nghiem truoc tien trong repo nay.

Doc them:

- `README_layered_guard_defense_vi.md`

---

## 3. Cac giai phap defense huu ich ben ngoai repo

Ben duoi la nhung huong co gia tri tham khao thuc su, khong chi la "liet ke cho du".

---

## 4. Nhom 1: Structured separation va provenance-based defense

Day la nhom quan trong nhat doi voi MedAgentSim.

### 4.1 Spotlighting

#### Paper Spotlighting

- <https://arxiv.org/abs/2403.14720>

#### Link tham khao Spotlighting

- Microsoft Research: <https://www.microsoft.com/en-us/research/publication/defending-against-indirect-prompt-injection-attacks-with-spotlighting/>

#### Y tuong Spotlighting

Khong de model nhin thay mot blob text bi tron lan. Thay vao do, bien doi input de tao "dau vet nguon goc" cho tung doan du lieu.

Cac kieu spotlighting thuong gap:

- delimiting
- data marking
- encoding / transformation

#### Uu diem Spotlighting

- hieu qua rat cao trong black-box setting
- paper bao cao giam ASR tu >50% xuong <2% trong mot so thuc nghiem
- rat phu hop voi bai toan content tu ben ngoai chen vao prompt

#### Nhuoc diem Spotlighting

- can prompt engineering can than
- khong phai model nao cung huong ung giong nhau

#### Muc do phu hop cua Spotlighting voi MedAgentSim

Rat cao.

Ly do:

- MedAgentSim hien dang noi patient / measurement content vao prompt doctor
- Spotlighting danh dung vao diem yeu cau truc nay

#### Khuyen nghi voi Spotlighting

- Nen hoc theo tinh than Spotlighting de xay prompt builder
- `layered_guard` da di theo huong nay

---

### 4.2 BIPIA: Boundary Awareness + Explicit Reminder

#### Paper BIPIA

- <https://arxiv.org/abs/2312.14197>

#### Repo BIPIA

- <https://github.com/microsoft/BIPIA>

#### Y tuong BIPIA

Paper chi ra 2 nguyen nhan chinh:

1. model khong phan biet duoc context thong tin va action instruction
2. model khong du nhan thuc rang instruction nam trong external content thi khong nen thuc thi

Hai defense black-box cua paper:

- boundary awareness
- explicit reminder

#### Uu diem BIPIA

- rat practical
- khong bat buoc fine-tune
- phu hop voi he thong co external content injection

#### Nhuoc diem BIPIA

- van phu thuoc prompt
- khong du neu attacker rat tinh vi va lap lai nhieu lan

#### Muc do phu hop cua BIPIA voi MedAgentSim

Rat cao.

#### Khuyen nghi voi BIPIA

- Nen doc ky paper nay neu ban muon nang cap prompt format them nua
- rat hop de cai dat nhanh trong code hien co

---

### 4.3 StruQ

#### Paper StruQ

- <https://arxiv.org/abs/2402.06363>

#### Repo StruQ

- <https://github.com/Sizhe-Chen/StruQ>

#### Y tuong StruQ

StruQ dua prompt va data vao 2 kenh co cau truc rieng. Model duoc train de chi nghe instruction o prompt channel va bo qua instruction trong data channel.

#### Uu diem StruQ

- manh hon prompting defense thong thuong
- y tuong "dung huong" ve mat kien truc

#### Nhuoc diem StruQ

- can model duoc train / fine-tune theo dang structured queries
- kho ap ngay vao codebase ma khong co ha tang white-box

#### Muc do phu hop cua StruQ voi MedAgentSim

Trung binh den cao.

Neu ban:

- chi muon giai phap dung ngay
  - qua nang
- muon nghien cuu luan van / nang cap sau
  - rat dang tham khao

#### Khuyen nghi voi StruQ

- Nen doc de lay y tuong thiet ke
- chua can implement full trong giai doan benchmark nhanh

---

### 4.4 SecAlign

#### Repo SecAlign

- <https://github.com/facebookresearch/SecAlign>

#### Tong quan SecAlign

SecAlign la huong tiep noi sau StruQ, dua prompt injection defense vao bai toan preference optimization.

#### Uu diem SecAlign

- bao cao kha manh trong tong quat hoa attack
- ve ly thuyet tot hon prompt-only defense

#### Nhuoc diem SecAlign

- can fine-tune / training pipeline
- khong phai lua chon nhe de dua vao MedAgentSim ngay lap tuc

#### Muc do phu hop cua SecAlign voi MedAgentSim

Cao ve mat nghien cuu, trung binh ve mat trien khai ngay.

---

## 5. Nhom 2: Detector chuyen dung

Detector chuyen dung nen la lop bo sung, khong nen thay the architecture defense.

### 5.1 Prompt Guard 2

#### Tai lieu Prompt Guard 2

- <https://llama.meta.com/docs/model-cards-and-prompt-formats/prompt-guard/>

#### Repo lien quan cua Prompt Guard 2

- Purple Llama: <https://github.com/meta-llama/PurpleLlama>

#### Y tuong Prompt Guard 2

BERT classifier nho de phat hien prompt injection / jailbreak.

#### Uu diem Prompt Guard 2

- nhe hon viec goi 1 LLM lon lam detector
- nhanh
- thuc dung

#### Nhuoc diem Prompt Guard 2

- context window co han
- van co false positive / false negative
- khong tu giai quyet viec instruction-data separation

#### Muc do phu hop cua Prompt Guard 2 voi MedAgentSim

Cao.

#### Khuyen nghi voi Prompt Guard 2

- rat nen dung nhu lop pre-filter
- neu ban muon detector chuyen dung, day la mot lua chon dep

---

### 5.2 ProtectAI LLM Guard

#### Repo ProtectAI LLM Guard

- <https://github.com/protectai/llm-guard>

#### Tai lieu ProtectAI LLM Guard

- <https://protectai.github.io/llm-guard/input_scanners/prompt_injection/>

#### Y tuong ProtectAI LLM Guard

Thu vien scanner cho prompt injection, PII, harmful language, va cac input risk khac.

#### Uu diem ProtectAI LLM Guard

- de nhung vao Python app
- scanner prompt injection co san
- phu hop pipeline engineering

#### Nhuoc diem ProtectAI LLM Guard

- chua phai architecture-level defense
- can tuning threshold

#### Muc do phu hop cua ProtectAI LLM Guard voi MedAgentSim

Cao.

#### Khuyen nghi voi ProtectAI LLM Guard

- rat hop neu ban muon mot detector "drop-in"
- `layered_guard` hien dang dung huong tuong tu qua Hugging Face classifier

---

### 5.3 InjecGuard

#### Paper / thong tin InjecGuard

- <https://arxiv.org/html/2410.22770v2>

#### Y tuong InjecGuard

Tap trung vao van de over-defense, nghia la detector chan nham qua nhieu benign inputs.

#### Uu diem InjecGuard

- huu ich neu ban muon benchmark detector mot cach nghiem tuc
- nhac nho rang FPR cung rat quan trong

#### Nhuoc diem InjecGuard

- khong phai framework production hoan chinh cho MedAgentSim

#### Muc do phu hop cua InjecGuard voi MedAgentSim

Trung binh.

#### Khuyen nghi voi InjecGuard

- Nen doc neu ban muon toi uu detector threshold va dataset danh gia.

---

## 6. Nhom 3: Framework guardrails

### 6.1 NVIDIA NeMo Guardrails

#### Repo NVIDIA NeMo Guardrails

- <https://github.com/NVIDIA/NeMo-Guardrails>

#### Paper / docs cua NVIDIA NeMo Guardrails

- <https://research.nvidia.com/publication/2023-10_nemo-guardrails-toolkit-controllable-and-safe-llm-applications-programmable>
- <https://docs.nvidia.com/nemo/guardrails/introduction.html>

#### Y tuong NVIDIA NeMo Guardrails

Dat mot lop dialog / moderation / safety rails giua ung dung va LLM.

#### Uu diem NVIDIA NeMo Guardrails

- framework lon, ro rang
- hop cho he thong hoi thoai nhieu luot
- co support evaluation va vulnerability scanning

#### Nhuoc diem NVIDIA NeMo Guardrails

- nang hon so voi viec sua truc tiep trong codebase
- can hoc them framework
- co the "qua tay" neu muc tieu chi la benchmark prompt injection tren MedAgentSim

#### Muc do phu hop cua NVIDIA NeMo Guardrails voi MedAgentSim

Trung binh den cao.

#### Khuyen nghi voi NVIDIA NeMo Guardrails

- Phu hop neu ban muon bien MedAgentSim thanh mot he thong guardrailed nghiem tuc hon
- Chua phai lua chon nhe nhat de thu nhanh

---

## 7. Nhom 4: By-design / capability-based defense

### 7.1 CaMeL

#### Paper CaMeL

- <https://arxiv.org/abs/2503.18813>

#### Repo CaMeL

- <https://github.com/google-research/camel-prompt-injection>

#### Y tuong CaMeL

Bao ve agent bang cach tach control flow va data flow, va gan capability / policy cho cac gia tri va tool call.

#### Uu diem CaMeL

- rat manh ve mat kien truc
- huong "secure by design"
- rat hop cho agent co tool nguy hiem

#### Nhuoc diem CaMeL

- kho cai dat vao code co san neu he thong chua duoc thiet ke theo capability model
- vuot qua nhu cau benchmark prompt injection co ban

#### Muc do phu hop cua CaMeL voi MedAgentSim

Trung binh hien tai, nhung se cao hon neu sau nay MedAgentSim:

- goi API ben ngoai
- ghi file
- thao tac he thong that

#### Khuyen nghi voi CaMeL

- Nen doc de dinh huong kien truc lau dai
- Chua can la buoc dau tien

---

## 8. Nhom 5: Nguyen tac thuc chien va checklist

### 8.1 OWASP Prompt Injection Prevention Cheat Sheet

#### Link OWASP cheat sheet

- <https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html>

#### Gia tri cua OWASP cheat sheet

Khong phai paper, nhung rat huu ich vi tong hop:

- input filtering
- structured prompt
- output validation
- HITL
- least privilege
- monitoring

#### Muc do phu hop cua OWASP cheat sheet voi MedAgentSim

Rat cao.

#### Khuyen nghi voi OWASP cheat sheet

- Nen xem day la checklist thuc thi
- dung de doi chieu xem code hien tai con thieu lop nao

---

## 9. Bang tong hop khuyen nghi cho MedAgentSim

| Giai phap | Loai | Paper / repo | Do manh ky vong | Do kho trien khai | Phu hop voi MedAgentSim |
| --- | --- | --- | --- | --- | --- |
| `known_answer` | Detector don gian | co san trong repo | Thap | Rat de | Thap |
| `llm_based` | LLM classifier | co san trong repo | Trung binh | De | Trung binh |
| `layered_guard` | Structured + detector + output validation | co san trong repo | Trung binh den cao | Trung binh | Rat cao |
| Spotlighting | Structured prompting | paper | Cao | Trung binh | Rat cao |
| BIPIA black-box defense | Boundary awareness | paper + repo | Cao | Trung binh | Rat cao |
| Prompt Guard 2 | Detector chuyen dung | docs + repo | Trung binh | De | Cao |
| ProtectAI LLM Guard | Detector / scanner | repo | Trung binh | De | Cao |
| StruQ | White-box / structured query training | paper + repo | Cao | Cao | Trung binh den cao |
| SecAlign | Preference optimization defense | repo | Cao | Rat cao | Trung binh den cao |
| NeMo Guardrails | Framework guardrail | repo + paper | Trung binh den cao | Trung binh den cao | Trung binh den cao |
| CaMeL | By-design secure agent | paper + repo | Rat cao | Rat cao | Trung binh hien tai |

---

## 10. Nen chon giai phap nao?

### 10.1 Neu muc tieu la benchmark ngay trong codebase hien tai

Nen uu tien:

1. `layered_guard`
2. Prompt Guard 2 hoac LLM Guard lam detector plug-in
3. benchmark lai bang cung matrix

### 10.2 Neu muc tieu la viet luan van / nghien cuu nghiem tuc hon

Nen tham khao sau:

1. Spotlighting
2. BIPIA
3. StruQ
4. SecAlign

### 10.3 Neu muc tieu la he thong production co tool nguy hiem

Nen bo sung:

1. least privilege
2. capability control
3. audit log
4. human-in-the-loop
5. framework guardrails hoac huong CaMeL

---

## 11. Khuyen nghi cu the cho MedAgentSim

Toi khuyen nghi lo trinh sau:

1. Dung `layered_guard` lam defense chinh de test ngay.
2. Chay benchmark cong bang giua:
   - `none`
   - `known_answer`
   - `llm_based`
   - `layered_guard`
3. Neu `layered_guard` cho ket qua tot nhung van con lo:
   - doi detector sang Prompt Guard 2
   - hoac scanner tu `llm-guard`
4. Neu muon nang cap theo huong nghien cuu:
   - hoc theo Spotlighting / BIPIA black-box defense
5. Neu co dieu kien fine-tune:
   - xem StruQ / SecAlign

---

## 12. Link tham khao nhanh

### Giup trien khai ngay

- OWASP cheat sheet: <https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html>
- LLM Guard: <https://github.com/protectai/llm-guard>
- Prompt Guard 2 docs: <https://llama.meta.com/docs/model-cards-and-prompt-formats/prompt-guard/>
- NeMo Guardrails: <https://github.com/NVIDIA/NeMo-Guardrails>

### Giup thiet ke prompt / architecture

- Spotlighting: <https://arxiv.org/abs/2403.14720>
- BIPIA: <https://arxiv.org/abs/2312.14197>
- StruQ: <https://arxiv.org/abs/2402.06363>

### Giup nghien cuu defense manh hon

- SecAlign: <https://github.com/facebookresearch/SecAlign>
- CaMeL: <https://arxiv.org/abs/2503.18813>

---

## 13. Ket luan cuoi cung

Neu hoi "giai phap defense nao hieu qua cho MedAgentSim?", cau tra loi ngan gon la:

- khong nen tin vao mot detector don le
- huong hieu qua nhat la tach instruction va data + detector bo sung + output validation

Vi vay, trong boi canh codebase nay:

- `known_answer` la baseline yeu
- `llm_based` la baseline kha hon nhung chua du
- `layered_guard` la lua chon hop ly nhat de thu nghiem ngay
- ve mat hoc thuat, Spotlighting va BIPIA la 2 huong nen doc ky nhat
- neu muon white-box defense manh hon, xem StruQ va SecAlign
