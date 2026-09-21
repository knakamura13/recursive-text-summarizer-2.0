import { expect, test } from '@playwright/test';

test('empty library shows import prompt', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('heading', { name: 'Import a document to begin' })).toBeVisible();
	await expect(page.getByRole('button', { name: 'Import documents' })).toBeVisible();
});

test('settings page loads', async ({ page }) => {
	await page.goto('/settings');
	await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
});
