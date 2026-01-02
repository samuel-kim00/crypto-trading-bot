# 🚀 GitHub 업로드 가이드

## 1단계: 파일 확인 및 커밋

### 현재 상태 확인
```bash
git status
```

### 모든 파일 추가
```bash
git add .
```

### 첫 커밋 생성
```bash
git commit -m "Initial commit: Ultra-aggressive scalping trading bot"
```

## 2단계: GitHub 저장소 생성

1. GitHub.com에 로그인
2. 우측 상단 **"+"** 버튼 클릭 → **"New repository"**
3. 저장소 이름 입력 (예: `crypto-trading-bot`)
4. **Public** 또는 **Private** 선택
5. **"Create repository"** 클릭
   - ⚠️ **"Initialize with README"** 체크하지 마세요!

## 3단계: GitHub에 업로드

### 저장소 URL 확인 후 실행
```bash
# 저장소 URL을 복사한 후 (예: https://github.com/yourusername/crypto-trading-bot.git)
git remote add origin https://github.com/yourusername/crypto-trading-bot.git
git branch -M main
git push -u origin main
```

## 4단계: 인증 (필요한 경우)

### Personal Access Token 사용
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. **"Generate new token"** 클릭
3. 권한 선택: `repo` (전체 저장소 권한)
4. 토큰 생성 후 복사
5. 비밀번호 입력 시 토큰 사용

### 또는 SSH 키 사용
```bash
# SSH 키가 있다면
git remote set-url origin git@github.com:yourusername/crypto-trading-bot.git
git push -u origin main
```

## 📋 업로드 전 체크리스트

- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
- [ ] API 키가 코드에 하드코딩되지 않았는지 확인
- [ ] 민감한 정보가 README에 없는지 확인
- [ ] `config/strategy_config.json`에 실제 API 키가 없는지 확인

## 🔒 보안 주의사항

### 절대 업로드하면 안 되는 파일:
- `.env` 파일
- API 키가 포함된 설정 파일
- 실제 거래 내역 (개인정보)
- 로그 파일 (민감한 정보 포함 가능)

### 안전하게 업로드되는 파일:
- ✅ 소스 코드
- ✅ `config/strategy_config.json.example` (예시 파일)
- ✅ README.md
- ✅ requirements.txt
- ✅ .gitignore

## 🎯 다음 단계

업로드 후:
1. GitHub 저장소에 README가 표시되는지 확인
2. Issues 탭에서 버그 리포트 받기
3. Releases 탭에서 버전 관리
4. Actions 탭에서 CI/CD 설정 (선택사항)

## 💡 유용한 Git 명령어

```bash
# 변경사항 확인
git status

# 변경사항 커밋
git add .
git commit -m "설명 메시지"

# GitHub에 푸시
git push

# 최신 변경사항 가져오기
git pull

# 브랜치 확인
git branch

# 커밋 히스토리
git log --oneline
```

---

**문제가 있으면 GitHub Issues에 등록하세요!** 🐛

