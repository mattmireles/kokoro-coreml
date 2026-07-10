#!/usr/bin/env node
/* global console, process */
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.cwd();
const args = new Set(process.argv.slice(2));
const strict = args.has('--strict');
const writeCoverage = args.has('--write-coverage');
const today = new Date().toISOString().slice(0, 10);

const read = (file) => fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '';
const exists = (file) => fs.existsSync(path.join(repoRoot, file));
const rel = (file) => path.relative(repoRoot, file).split(path.sep).join('/');

function walk(dir) {
  const root = path.join(repoRoot, dir);
  if (!fs.existsSync(root)) return [];
  const out = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) out.push(...walk(rel(full)));
    else out.push(rel(full));
  }
  return out;
}

function canonicalSources() {
  const docs = [
    ...walk('README'),
    ...['README.md', 'AGENTS.md', 'CLAUDE.md'].filter(exists),
  ].filter((file) => file.endsWith('.md'));
  return [...new Set(docs)].filter((file) => {
    if (file.startsWith('README/Wiki/')) return false;
    if (file.startsWith('README/Plans/')) return false;
    if (file.startsWith('README/Templates/')) return false;
    if (file.startsWith('README/Skills/')) return false;
    if (file.startsWith('README/Experiments/')) return false;
    return true;
  }).sort();
}

function wikiFiles() {
  return walk('README/Wiki').filter((file) => file.endsWith('.md')).sort();
}

function writeCoverageIndex(sources) {
  const target = path.join(repoRoot, 'README/Wiki/canonical-source-coverage.md');
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const linkFor = (source) => source.startsWith('README/')
    ? `../${source.slice('README/'.length)}`
    : `../../${source}`;
  const body = [
    '---',
    'title: Canonical Source Coverage',
    `last_synced: ${today}`,
    'coverage_index: true',
    '---',
    '',
    '# Canonical Source Coverage',
    '',
    'This is a deterministic coverage index, not a current-belief page. It lists readable repo-memory sources so `memory-health` can prove that the wiki layer has a path back to canonical docs.',
    '',
    '## Sources',
    '',
    ...sources.map((source) => `- [${source}](${linkFor(source)})`),
    '',
  ].join('\n');
  fs.writeFileSync(target, body);
}

function currentBeliefPages(files) {
  return files.filter((file) => {
    if (file.endsWith('/README.md')) return false;
    if (file.endsWith('/canonical-source-coverage.md')) return false;
    const text = read(path.join(repoRoot, file));
    if (/coverage_index:\s*true/.test(text)) return false;
    return /current[- ]belief|Current belief|claims:/i.test(text);
  });
}

const sources = canonicalSources();
if (writeCoverage) writeCoverageIndex(sources);
const wiki = wikiFiles();
const wikiText = wiki.map((file) => read(path.join(repoRoot, file))).join('\n');
const covered = sources.filter((source) => wikiText.includes(source));
const currentPages = currentBeliefPages(wiki);

const staleCurrentPages = currentPages.filter((file) => {
  const match = read(path.join(repoRoot, file)).match(/last_synced:\s*(\d{4}-\d{2}-\d{2})/);
  if (!match) return true;
  const ageDays = (Date.now() - Date.parse(`${match[1]}T00:00:00Z`)) / 86400000;
  return ageDays > 90;
});

const guidePath = ['README/Guides/In-Repo-memory-guide.md', 'README/guides/In-Repo-memory-guide.md']
  .find(exists);
const guideText = guidePath ? read(path.join(repoRoot, guidePath)) : '';
const routingText = ['AGENTS.md', 'CLAUDE.md', 'README.md']
  .filter(exists)
  .map((file) => read(path.join(repoRoot, file)))
  .join('\n');
const noteTemplateText = ['README/Templates/Notes-template.md', 'README/Templates/notes-template.md']
  .filter(exists)
  .map((file) => read(path.join(repoRoot, file)))
  .join('\n');
const planTemplateText = ['README/Templates/Plans-template.md', 'README/Templates/plan-template.md']
  .filter(exists)
  .map((file) => read(path.join(repoRoot, file)))
  .join('\n');

const checks = {
  canonical_sources: sources.length,
  covered_sources: covered.length,
  uncovered_sources: sources.length - covered.length,
  coverage_percent: sources.length ? Number(((covered.length / sources.length) * 100).toFixed(1)) : 100,
  wiki_files: wiki.length,
  current_belief_pages: currentPages.length,
  stale_current_pages: staleCurrentPages.length,
  has_memory_guide: Boolean(guidePath),
  routing_mentions_wiki: /README\/Wiki/.test(routingText),
  routing_mentions_memory_health: /memory-health/.test(routingText),
  guide_rejects_vector_db_default: /vector DB is not the default|No vector DB/i.test(guideText),
  guide_requires_sources: /sources?:|provenance/i.test(guideText),
  notes_template_executable_memory: /Executable Memory|Regression test:|Not testable:/i.test(noteTemplateText),
  plans_template_executable_memory: /Executable Memory|Regression test:|Not testable:/i.test(planTemplateText),
};

const failures = [];
if (checks.coverage_percent < 100) failures.push('canonical source coverage is below 100%');
if (checks.current_belief_pages < 3) failures.push('fewer than 3 current-belief wiki pages');
if (checks.stale_current_pages > 0) failures.push('stale or undated current-belief wiki pages');
if (!checks.has_memory_guide) failures.push('missing in-repo memory guide');
if (!checks.routing_mentions_wiki) failures.push('routing docs do not mention README/Wiki');
if (!checks.routing_mentions_memory_health) failures.push('routing docs do not mention memory-health');
if (!checks.guide_rejects_vector_db_default) failures.push('memory guide does not reject vector DB as default');
if (!checks.guide_requires_sources) failures.push('memory guide does not require provenance');
if (!checks.notes_template_executable_memory) failures.push('notes template lacks executable-memory field');
if (!checks.plans_template_executable_memory) failures.push('plans template lacks executable-memory field');

const grade = failures.length === 0 ? 'A' : failures.length <= 2 ? 'B' : failures.length <= 4 ? 'C' : failures.length <= 6 ? 'D' : 'F';
console.log(JSON.stringify({ grade, checks, failures }, null, 2));
if (strict && failures.length > 0) process.exit(1);
