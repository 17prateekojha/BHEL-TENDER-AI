import * as vscode from 'vscode';

interface Source { file: string; abs_path: string; loc: string; start_line: number; score: number; }
interface Snippet { file: string; abs_path: string; language: string; start_line: number; code: string; }
interface Reply { refused: boolean; markdown: string; body: string; sources: Source[]; snippets: Snippet[]; }

function serverUrl(): string {
  return vscode.workspace.getConfiguration('ragAssistant').get<string>('serverUrl', 'http://127.0.0.1:8000');
}

async function post(path: string, payload?: unknown): Promise<any> {
  const res = await fetch(`${serverUrl()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload ? JSON.stringify(payload) : undefined,
  });
  if (!res.ok) { throw new Error(`Server returned ${res.status}`); }
  return res.json();
}

async function ask(query: string): Promise<Reply | undefined> {
  try {
    return await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: 'Searching documents...' },
      () => post('/ask', { query }) as Promise<Reply>
    );
  } catch (err) {
    vscode.window.showErrorMessage(
      `Cannot reach the RAG server at ${serverUrl()}. Start it with: uvicorn server:app --port 8000 (${err})`
    );
    return undefined;
  }
}

// Markdown links of the form file:///path#L12 open the file at that line from the preview.
function sourceLink(s: Source): string {
  const uri = vscode.Uri.file(s.abs_path).with({ fragment: `L${s.start_line}` });
  return `- [\`${s.file}\`](${uri.toString()}) (${s.loc})`;
}

async function showAnswer(query: string, r: Reply) {
  let md = `# ${query}\n\n${r.body}`;
  if (r.sources.length) {
    md += `\n\n## Sources\n${r.sources.map(sourceLink).join('\n')}`;
  }
  const doc = await vscode.workspace.openTextDocument({ language: 'markdown', content: md });
  await vscode.window.showTextDocument(doc, { preview: false });
  await vscode.commands.executeCommand('markdown.showPreviewToSide');
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand('ragAssistant.ask', async () => {
      const query = await vscode.window.showInputBox({
        prompt: 'Ask a question about your documents', ignoreFocusOut: true,
      });
      if (!query) { return; }
      const r = await ask(query);
      if (r) { await showAnswer(query, r); }
    }),

    vscode.commands.registerCommand('ragAssistant.insertCode', async () => {
      const query = await vscode.window.showInputBox({
        prompt: 'Describe the code you need (e.g. "login function")', ignoreFocusOut: true,
      });
      if (!query) { return; }
      const r = await ask(query);
      if (!r) { return; }
      if (r.refused) { vscode.window.showWarningMessage(r.body); return; }
      if (!r.snippets.length) {
        vscode.window.showInformationMessage('No relevant code was found in the documents.');
        return;
      }
      const pick = await vscode.window.showQuickPick(
        r.snippets.map(s => ({
          label: `${s.file}:${s.start_line}`,
          description: s.language,
          detail: s.code.split('\n')[0].slice(0, 100),
          snippet: s,
        })),
        { placeHolder: 'Select a snippet to insert at the cursor' }
      );
      const editor = vscode.window.activeTextEditor;
      if (pick && editor) {
        await editor.edit(b => b.replace(editor.selection, pick.snippet.code));
      } else if (pick) {
        await vscode.env.clipboard.writeText(pick.snippet.code);
        vscode.window.showInformationMessage('No open editor: snippet copied to clipboard.');
      }
    }),

    vscode.commands.registerCommand('ragAssistant.reindex', async () => {
      try {
        const out = await post('/reindex');
        vscode.window.showInformationMessage(`Indexed ${out.files} files (${out.chunks} chunks).`);
      } catch (err) {
        vscode.window.showErrorMessage(`Re-index failed: ${err}`);
      }
    })
  );
}

export function deactivate() {}
