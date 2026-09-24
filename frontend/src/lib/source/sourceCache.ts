import { api } from '$lib/api/client';
import { LruCache, SourceChunkLoader } from './sourceModel';

// Document text never changes after its Import, so chunk caches outlive the viewer: switching
// tabs or panes (which remounts the viewer) reuses what was already fetched.
const loaders = new LruCache<string, SourceChunkLoader>(3);

export function sourceLoader(documentId: string): SourceChunkLoader {
	let loader = loaders.get(documentId);
	if (!loader) {
		loader = new SourceChunkLoader((offset, limit) => api.getSource(documentId, offset, limit));
		loaders.set(documentId, loader);
	}
	return loader;
}
