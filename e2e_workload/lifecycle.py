"""Pure cache-entry bookkeeping; wall-clock ages survive service downtime."""
import copy


class Lifecycle:
    def __init__(self, max_age_seconds=None):
        self.max_age_seconds = max_age_seconds
        self.entries = {}
        self.events = []

    def observe(self, resident, owner, inherited, now):
        """Observe actual objects, never infer writes from scheduled requests."""
        for key in list(self.entries):
            old = self.entries[key]
            if key not in resident or old['object_token'] != resident[key]['object_token']:
                self.events.append(dict(kind='evict', reason='capacity_or_native_removal',
                    key=key, at=now, age_seconds=max(0, now-old['created_at']), entry=copy.deepcopy(old)))
                del self.entries[key]
        for key, obj in resident.items():
            if key in self.entries:
                continue
            if owner is None:
                raise RuntimeError('Unattributed native cache admission')
            existing = [e['created_at'] for e in self.entries.values() if e['group'] == obj['group']]
            entry = dict(obj, created_at=min(existing) if existing else obj.get('admitted_at', now),
                         direct_request=owner['id'], direct_role=owner['role'],
                         sources=sorted(set(inherited) | {owner['id']}))
            self.entries[key] = entry
            self.events.append(dict(kind='store', key=key, at=now, entry=copy.deepcopy(entry)))

    def expired_groups(self, now):
        if self.max_age_seconds is None:
            return set()
        return {e['group'] for e in self.entries.values()
                if now-e['created_at'] >= self.max_age_seconds}

    def expire(self, resident, remove, now):
        groups = self.expired_groups(now)
        keys = [k for k, e in self.entries.items() if e['group'] in groups]
        # A new request must never see half-expired layers or pinned stale KV.
        if any(k in resident and not resident[k]['can_evict'] for k in keys):
            raise RuntimeError('Expired cache group is still in use at request boundary')
        for key in keys:
            entry = self.entries[key]
            if key in resident and not remove(key):
                raise RuntimeError('Native cache refused expiration')
            self.events.append(dict(kind='evict', reason='max_age', key=key, at=now,
                age_seconds=max(0, now-entry['created_at']), entry=copy.deepcopy(entry)))
            del self.entries[key]

    def loaded_metadata(self, keys, now):
        return [dict(key=k, age_seconds=max(0, now-self.entries[k]['created_at']),
                     **{n: v for n, v in self.entries[k].items() if n != 'object_token'})
                for k in keys if k in self.entries]

    def state_dict(self):
        return dict(max_age_seconds=self.max_age_seconds,
                    entries={k: {n: v for n, v in e.items() if n != 'object_token'}
                             for k, e in self.entries.items()})

    def restore(self, state, resident):
        if state['max_age_seconds'] != self.max_age_seconds or set(state['entries']) != set(resident):
            raise RuntimeError('Snapshot lifecycle/capacity mismatch')
        self.entries = {k: dict(e, object_token=resident[k]['object_token'])
                        for k, e in state['entries'].items()}
        self.events.clear()
