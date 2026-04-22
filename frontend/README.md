# Frontend (Next.js)

This frontend uploads a form file to the backend endpoint `POST /process-form`, then shows the generated PDF and optional mapping preview.

## Run locally

1. Install dependencies:

```bash
npm install
```

2. Configure environment:

```bash
copy .env.example .env.local
```

3. Start dev server:

```bash
npm run dev
```

By default, API requests go to `http://localhost:8000`.
