<script lang="ts" module>
	const FOCUSABLE =
		'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

	function focusables(root: HTMLElement): HTMLElement[] {
		return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
			(element) => !element.closest('[hidden], [inert]')
		);
	}
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		open = $bindable(false),
		title,
		description,
		onclose,
		dismissible = true,
		size = 'md',
		variant = 'dialog',
		children,
		footer
	}: {
		open?: boolean;
		title: string;
		description?: string;
		/** Called when the user dismisses the dialog (Escape, backdrop, close button). */
		onclose?: () => void;
		/** False while work is in flight: Escape, the backdrop, and the close button do nothing. */
		dismissible?: boolean;
		size?: 'sm' | 'md' | 'lg';
		/** `sheet`: full screen on phones, a side panel from 640 px. */
		variant?: 'dialog' | 'sheet';
		children: Snippet;
		footer?: Snippet;
	} = $props();

	const id = $props.id();
	let element: HTMLDialogElement | undefined = $state();
	let returnFocus: HTMLElement | null = null;
	let pressedBackdrop = false;

	$effect(() => {
		if (!element) return;
		if (open && !element.open) {
			returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
			element.showModal();
			const preferred =
				element.querySelector<HTMLElement>('[data-autofocus]') ??
				focusables(element).find((candidate) => !candidate.closest('.dialog-header'));
			(preferred ?? element).focus();
		} else if (!open && element.open) {
			element.close();
		}
	});

	function dismiss() {
		if (!dismissible) return;
		open = false;
		onclose?.();
	}

	function onCancel(event: Event) {
		// Escape: keep `open` the single source of truth.
		event.preventDefault();
		dismiss();
	}

	function onNativeClose() {
		if (open) open = false;
		if (returnFocus?.isConnected && returnFocus !== document.body) returnFocus.focus();
		else document.getElementById('main')?.focus();
		returnFocus = null;
	}

	function onKeydown(event: KeyboardEvent) {
		if (event.key !== 'Tab' || !element) return;
		const items = focusables(element);
		if (items.length === 0) {
			event.preventDefault();
			return;
		}
		const first = items[0];
		const last = items[items.length - 1];
		if (event.shiftKey && (document.activeElement === first || document.activeElement === element)) {
			event.preventDefault();
			last.focus();
		} else if (!event.shiftKey && document.activeElement === last) {
			event.preventDefault();
			first.focus();
		}
	}
</script>

<!-- The dialog element itself is only the target of clicks on its backdrop; keyboard users close it with Escape. -->
<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<dialog
	bind:this={element}
	class="dialog {size} {variant}"
	tabindex="-1"
	aria-labelledby="{id}-title"
	aria-describedby={description ? `${id}-description` : undefined}
	oncancel={onCancel}
	onclose={onNativeClose}
	onkeydown={onKeydown}
	onpointerdown={(event) => (pressedBackdrop = event.target === element)}
	onclick={(event) => {
		if (pressedBackdrop && event.target === element) dismiss();
		pressedBackdrop = false;
	}}
>
	{#if open}
		<div class="surface">
			<header class="dialog-header">
				<h2 id="{id}-title">{title}</h2>
				<button
					type="button"
					class="button ghost icon"
					aria-label="Close"
					disabled={!dismissible}
					onclick={dismiss}>×</button
				>
			</header>
			<div class="dialog-body">
				{#if description}
					<p id="{id}-description" class="description">{description}</p>
				{/if}
				{@render children()}
			</div>
			{#if footer}
				<footer class="dialog-footer">{@render footer()}</footer>
			{/if}
		</div>
	{/if}
</dialog>

<style>
	.dialog {
		padding: 0;
		border: 1px solid var(--color-border);
		border-radius: var(--radius-lg);
		background: var(--color-surface-elevated);
		color: var(--color-text);
		box-shadow: var(--shadow-lg);
		width: min(34rem, calc(100vw - 1rem));
		max-width: calc(100vw - 1rem);
		max-height: calc(100dvh - 1rem);
		overflow: hidden;
	}

	.dialog.sm {
		width: min(26rem, calc(100vw - 1rem));
	}

	.dialog.lg {
		width: min(48rem, calc(100vw - 1rem));
	}

	.dialog::backdrop {
		background: var(--color-overlay);
	}

	.surface {
		display: flex;
		flex-direction: column;
		max-height: calc(100dvh - 1rem);
	}

	.dialog-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-3);
		padding: var(--space-2) var(--space-2) var(--space-2) var(--space-5);
		border-bottom: 1px solid var(--color-border);
	}

	.dialog-header h2 {
		font-size: 1.0625rem;
		min-width: 0;
	}

	.dialog-body {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
		padding: var(--space-5);
		overflow: auto;
		overscroll-behavior: contain;
		min-height: 0;
	}

	.description {
		color: var(--color-text-muted);
	}

	.dialog-footer {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: var(--space-3);
		padding: var(--space-3) var(--space-5);
		border-top: 1px solid var(--color-border);
	}

	/* Sheet: full screen on phones, a panel on the right from 640 px. */
	.dialog.sheet {
		width: 100vw;
		max-width: 100vw;
		height: 100dvh;
		max-height: 100dvh;
		margin: 0;
		border: none;
		border-radius: 0;
	}

	.dialog.sheet .surface {
		height: 100%;
		max-height: 100dvh;
	}

	.dialog.sheet .dialog-body {
		flex: 1;
	}

	@media (min-width: 640px) {
		.dialog.sheet {
			width: min(30rem, 90vw);
			margin-left: auto;
			border-left: 1px solid var(--color-border);
		}
	}
</style>
