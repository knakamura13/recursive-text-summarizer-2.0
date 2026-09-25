import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		// SPA: FastAPI serves index.html for every non-API route.
		adapter: adapter({
			fallback: 'index.html'
		}),
		typescript: {
			// Type-check the Playwright specs and config with the app.
			config: (tsconfig) => {
				tsconfig.include.push('../e2e/**/*.ts', '../playwright.config.ts');
			}
		}
	}
};

export default config;
