import { useState } from 'react';
import { VizPanel, Segmented, Button, Legend, Stats, Note } from './Viz';
import { bySlug } from './curriculumRoutes';

type SystemId = 'postgres' | 'dynamo' | 'kafka';
type Q = {
  id: 'durability' | 'atomicity' | 'replication' | 'reads' | 'failure';
  question: string;
  options: { id: string; text: string }[];
  answer: Record<SystemId, string>;
  why: Record<SystemId, string>;
  modules: string[];
};

const SYSTEMS: { value: SystemId; label: string }[] = [
  { value: 'postgres', label: 'PostgreSQL' },
  { value: 'dynamo', label: 'DynamoDB' },
  { value: 'kafka', label: 'Kafka' },
];

const QUESTIONS: Q[] = [
  {
    id: 'durability',
    question: '1. When is an acknowledged write safe?',
    options: [
      { id: 'mem', text: "Once it is in the leader's memory" },
      { id: 'local', text: 'Once a log record is flushed to local disk' },
      { id: 'quorum', text: 'Once a quorum of replicas has persisted it' },
      { id: 'isr', text: 'Once every in-sync replica has it' },
    ],
    answer: { postgres: 'local', dynamo: 'quorum', kafka: 'isr' },
    why: {
      postgres: 'With the default synchronous_commit = on, COMMIT returns after the commit record is flushed to the local WAL. Synchronous replication can additionally wait for a standby, but that is opt-in.',
      dynamo: "Each partition is a replication group; per the 2022 DynamoDB paper, a write is acknowledged once a quorum of that group's replicas has persisted the log record, and the replicas sit in different availability zones.",
      kafka: 'With acks=all, a record is committed once every replica in the current in-sync replica set has it; if that set shrinks below min.insync.replicas, such writes are rejected rather than accepted with less protection. Kafka relies on replication rather than an fsync per record for durability.',
    },
    modules: ['hardware-os-durability', 'write-ahead-logging-and-recovery'],
  },
  {
    id: 'atomicity',
    question: '2. What succeeds or fails together?',
    options: [
      { id: 'one', text: 'A single row, item or record' },
      { id: 'txn-node', text: 'A transaction spanning many rows on one node' },
      { id: 'txn-api', text: 'Many items or partitions, but only through an explicit transaction API' },
      { id: 'default-multi', text: 'Many items or partitions, by default' },
    ],
    answer: { postgres: 'txn-node', dynamo: 'txn-api', kafka: 'txn-api' },
    why: {
      postgres: 'A transaction can touch any number of rows and tables, and commits or aborts as one — on one server.',
      dynamo: 'A single-item write is atomic on its own. Changing several items atomically requires TransactWriteItems, which runs a commit protocol across them.',
      kafka: 'Without transactions, each record lands in one partition independently, so a batch spanning several partitions can partially succeed. Kafka transactions make writes to several partitions — and the consumer offsets that go with them — commit or abort together.',
    },
    modules: ['transactions-and-isolation', 'distributed-transactions'],
  },
  {
    id: 'replication',
    question: '3. What is the unit of replication and partitioning?',
    options: [
      { id: 'instance', text: 'The whole database instance, shipped as one log' },
      { id: 'partition', text: 'A partition — each one is replicated on its own' },
      { id: 'row', text: 'Individual rows, each replicated separately' },
    ],
    answer: { postgres: 'instance', dynamo: 'partition', kafka: 'partition' },
    why: {
      postgres: "Streaming replication ships the entire server's WAL to a standby. Core PostgreSQL does not shard: declarative partitioning splits a table within one server, and spreading data across servers takes an extension such as Citus.",
      dynamo: 'Items are placed into partitions by partition key, and each partition is replicated as its own group with its own leader.',
      kafka: 'A topic is split into partitions, and each partition is its own replicated log with a leader and followers.',
    },
    modules: ['replication', 'partitioning-and-sharding'],
  },
  {
    id: 'reads',
    question: '4. What does a read see, and how stale can it be?',
    options: [
      { id: 'latest-anywhere', text: 'Always the latest committed write, from any node' },
      { id: 'snapshot', text: 'A consistent snapshot — but a replica may lag the primary' },
      { id: 'eventual-default', text: 'Eventually consistent by default, with a strongly consistent option' },
      { id: 'watermark', text: 'Only records every in-sync replica has (up to the high watermark)' },
    ],
    answer: { postgres: 'snapshot', dynamo: 'eventual-default', kafka: 'watermark' },
    why: {
      postgres: 'Every query reads an MVCC snapshot. On the primary that snapshot is current; on an asynchronous standby it can trail the primary by the replication lag.',
      dynamo: 'Reads are eventually consistent unless ConsistentRead is set on a base-table read. Global secondary indexes only support eventually consistent reads.',
      kafka: 'Consumers can only read up to the high watermark, the offset every in-sync replica has. With isolation.level=read_committed they also stop before records of transactions that are still open.',
    },
    modules: ['mvcc-storage-vacuum-gc', 'consistency-models'],
  },
  {
    id: 'failure',
    question: '5. What happens when the leader or primary fails?',
    options: [
      { id: 'nothing', text: 'Nothing is lost and no client notices' },
      { id: 'elect', text: 'A new leader is elected automatically from up-to-date replicas' },
      { id: 'external', text: 'Failover needs external tooling, and an asynchronous replica can lose recent commits' },
    ],
    answer: { postgres: 'external', dynamo: 'elect', kafka: 'elect' },
    why: {
      postgres: 'PostgreSQL does not promote a standby by itself; tools such as Patroni do. Promoting an asynchronous standby loses any commits that had not reached it yet.',
      dynamo: "The partition's replication group elects a new leader among its replicas, and in-flight requests may fail and need retrying.",
      kafka: 'The controller picks a new leader from the in-sync replicas. With unclean.leader.election.enable=false, the default, an out-of-sync replica is never chosen, so committed records survive as long as at least one in-sync replica does.',
    },
    modules: ['failure-detection-leases-fencing', 'consensus-and-coordination'],
  },
];

/** Reasoning checks across a learner's own answers for one system. */
function mismatches(a: Partial<Record<Q['id'], string>>): string[] {
  const out: string[] = [];
  if (a.atomicity === 'default-multi' && a.replication === 'partition')
    out.push('You said atomicity spans partitions by default, and that each partition replicates on its own. Both hold only if a commit protocol runs across partitions on every multi-partition write — Spanner and CockroachDB pay that cost on every such transaction. Check whether this system does.');
  if (a.durability === 'mem' && a.failure === 'nothing')
    out.push("You said a write is safe once it is in the leader's memory, and that a leader failure loses nothing. A write that only ever reached one node's memory cannot survive that node failing.");
  if (a.replication === 'instance' && (a.atomicity === 'txn-api' || a.atomicity === 'default-multi'))
    out.push('Your atomicity answer is about spanning partitions, but your replication answer says the whole instance is one log with no partitions to span.');
  if (a.reads === 'latest-anywhere' && a.failure === 'external')
    out.push('You said any node always returns the latest committed write, and that an asynchronous replica can be missing recent commits. A replica that can be behind cannot always return the latest write.');
  return out;
}

export default function FiveQuestionsCard() {
  const [system, setSystem] = useState<SystemId>('postgres');
  const [picks, setPicks] = useState<Record<SystemId, Partial<Record<Q['id'], string>>>>({ postgres: {}, dynamo: {}, kafka: {} });
  const [checked, setChecked] = useState<Record<SystemId, boolean>>({ postgres: false, dynamo: false, kafka: false });

  const mine = picks[system];
  const isChecked = checked[system];
  const answered = QUESTIONS.filter((q) => mine[q.id]).length;
  const right = QUESTIONS.filter((q) => mine[q.id] === q.answer[system]).length;
  const flags = mismatches(mine);

  const pick = (qid: Q['id'], oid: string) => {
    if (isChecked) return;
    setPicks((p) => ({ ...p, [system]: { ...p[system], [qid]: oid } }));
  };

  const moduleLink = (slug: string) => {
    const m = bySlug.get(slug);
    if (!m) return null;
    return m.url ? (
      <a key={slug} href={m.url}>
        {m.index} {m.title}
      </a>
    ) : (
      <span key={slug} title="Not written yet">
        {m.index} {m.title} (not written yet)
      </span>
    );
  };

  return (
    <VizPanel
      title="Five questions, three systems"
      subtitle="Answer the five questions for one system. The consistency check watches your answers as you go; Check reveals the real answers and the module that derives each one."
      controls={
        <>
          <Segmented label="System" value={system} onChange={setSystem} options={SYSTEMS} />
          <Button primary onClick={() => setChecked((c) => ({ ...c, [system]: true }))} disabled={isChecked || answered < QUESTIONS.length}>
            Check answers
          </Button>
          <Button
            onClick={() => {
              setPicks((p) => ({ ...p, [system]: {} }));
              setChecked((c) => ({ ...c, [system]: false }));
            }}
          >
            Clear
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Your answer', color: 'var(--viz-1)' },
            { label: '✓ Correct answer', color: 'var(--viz-good)' },
            { label: '✕ Your answer, not correct', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'System', value: SYSTEMS.find((s) => s.value === system)!.label },
            { label: 'Answered', value: `${answered} / ${QUESTIONS.length}` },
            { label: 'Correct', value: isChecked ? `${right} / ${QUESTIONS.length}` : '—' },
            { label: 'Consistency flags', value: flags.length },
          ]}
        />
      }
      note={
        <Note>
          {flags.length ? (
            <>
              <strong>⚠ Check these answers together.</strong> {flags[0]}
              {flags.length > 1 ? ` (${flags.length - 1} more below.)` : ''}
            </>
          ) : isChecked ? (
            <>
              <strong>
                {right} of {QUESTIONS.length} right for {SYSTEMS.find((s) => s.value === system)!.label}.
              </strong>{' '}
              Now switch systems and notice which answers change. Durability, atomicity and replication units move together — that is why these three systems behave so differently under failure.
            </>
          ) : (
            <>
              <strong>Pick an answer to every question.</strong> Each system keeps its own answers, so you can compare them side by side.
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Question</th>
              {SYSTEMS.map((s) => (
                <th key={s.value}>{s.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {QUESTIONS.map((q) => (
              <tr key={q.id}>
                <td>{q.question}</td>
                {SYSTEMS.map((s) => (
                  <td key={s.value}>{checked[s.value] ? q.options.find((o) => o.id === q.answer[s.value])!.text : 'check to reveal'}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'grid', gap: '0.85rem', minWidth: 320 }}>
        {QUESTIONS.map((q) => {
          const chosen = mine[q.id];
          const correct = q.answer[system];
          return (
            <fieldset key={q.id} style={{ border: '1px solid var(--viz-border)', borderRadius: 8, padding: '0.55rem 0.7rem', margin: 0 }}>
              <legend style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--viz-ink)', padding: '0 0.25rem' }}>{q.question}</legend>
              <div style={{ display: 'grid', gap: '0.3rem' }}>
                {q.options.map((o) => {
                  const isMine = chosen === o.id;
                  const isRight = isChecked && o.id === correct;
                  const isWrongPick = isChecked && isMine && o.id !== correct;
                  const edge = isRight ? 'var(--viz-good)' : isWrongPick ? 'var(--viz-critical)' : isMine ? 'var(--viz-1)' : 'transparent';
                  return (
                    <button
                      key={o.id}
                      type="button"
                      aria-pressed={isMine}
                      disabled={isChecked}
                      onClick={() => pick(q.id, o.id)}
                      style={{
                        textAlign: 'left',
                        fontSize: '0.78rem',
                        borderLeft: `4px solid ${edge}`,
                        opacity: isChecked && !isMine && !isRight ? 0.55 : 1,
                        cursor: isChecked ? 'default' : 'pointer',
                      }}
                    >
                      {isRight ? '✓ ' : isWrongPick ? '✕ ' : ''}
                      {o.text}
                    </button>
                  );
                })}
              </div>
              {isChecked ? (
                <p style={{ margin: '0.45rem 0 0', fontSize: '0.76rem', color: 'var(--viz-ink-2)' }}>
                  {q.why[system]} <strong style={{ color: 'var(--viz-ink)' }}>Derived in:</strong>{' '}
                  {q.modules.map((s, n) => (
                    <span key={s}>
                      {n ? ' · ' : ''}
                      {moduleLink(s)}
                    </span>
                  ))}
                </p>
              ) : null}
            </fieldset>
          );
        })}
        {flags.length > 1 ? (
          <ul style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.76rem', color: 'var(--viz-ink-2)' }}>
            {flags.slice(1).map((f) => (
              <li key={f}>⚠ {f}</li>
            ))}
          </ul>
        ) : null}
      </div>
    </VizPanel>
  );
}
