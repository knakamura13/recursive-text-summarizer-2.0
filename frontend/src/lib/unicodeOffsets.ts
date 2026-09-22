/** Convert Python code-point offsets to UTF-16 indices for DOM highlighting. */

export function pythonOffsetToUtf16Index(text: string, pythonOffset: number): number {
	const codePoints = Array.from(text);
	const prefix = codePoints.slice(0, pythonOffset).join('');
	return prefix.length;
}

export function sliceByPythonOffsets(text: string, start: number, end: number): string {
	const codePoints = Array.from(text);
	return codePoints.slice(start, end).join('');
}
