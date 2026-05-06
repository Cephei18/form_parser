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

Set `NEXT_PUBLIC_API_BASE_URL` to your backend URL, for example `http://127.0.0.1:8000` during local development or your EC2 load balancer / instance URL in deployment.
