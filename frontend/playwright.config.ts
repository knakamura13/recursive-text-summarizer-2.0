import { defineConfig, devices } from '@playwright/test';

// UI flows run against the Vite dev server with every /api/v1 call answered
// by page.route mocks (e2e/support/mockApi.ts); no Python backend is needed.
// Each frontend slice runs on its own port: E2E_PORT (default 5174).
const PORT = Number(process.env.E2E_PORT ?? 5174);

export default defineConfig({
	testDir: 'e2e',
	fullyParallel: true,
	forbidOnly: !!process.env.CI,
	retries: process.env.CI ? 1 : 0,
	reporter: process.env.CI ? 'line' : 'list',
	use: {
		baseURL: `http://127.0.0.1:${PORT}`,
		trace: 'retain-on-failure'
	},
	projects: [
		{ name: 'desktop-chromium', use: { ...devices['Desktop Chrome'] } },
		{ name: 'desktop-firefox', use: { ...devices['Desktop Firefox'] } },
		{ name: 'desktop-webkit', use: { ...devices['Desktop Safari'] } },
		{ name: 'iphone-13', use: { ...devices['iPhone 13'] } },
		{ name: 'ipad-gen-7', use: { ...devices['iPad (gen 7)'] } }
	],
	webServer: {
		command: `pnpm exec vite dev --host 127.0.0.1 --port ${PORT} --strictPort`,
		url: `http://127.0.0.1:${PORT}`,
		reuseExistingServer: !process.env.CI,
		timeout: 120_000
	}
});
