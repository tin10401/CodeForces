template<typename T, typename I = ll, typename II = ll, typename F = function<T(const T, const T)>, typename G = function<void(int i, int left, int right, I)>>
class SGT { 
    public: 
    int n;  
    vt<T> root;
	vt<II> lazy;
    T DEFAULT;
    F func;
    G apply_func;
	SGT(int n, T DEFAULT, F func, G apply_func = [](int i, int left, int right, I val){}) : func(func), apply_func(apply_func) {    
        this->n = n;
        this->DEFAULT = DEFAULT;
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1, DEFAULT);    
        lazy.rsz(k << 1, 0);
    }
    
    void update_at(int id, T val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, T val) {  
		pushDown;
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = func(root[lc], root[rc]);
    }

    void update_range(int start, int end, I val) { 
        update_range(entireTree, start, end, val);
    }
    
    void update_range(iter, int start, int end, I val) {    
        pushDown;   
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
			apply(i, left, right, val);
            pushDown;   
            return;
        }
        int middle = midPoint;  
        update_range(lp, start, end, val);    
        update_range(rp, start, end, val);    
        root[i] = func(root[lc], root[rc]);
    }

	void apply(iter, I val) {
        root[i] += val * (right - left + 1);
        lazy[i] += val;
    }

    void push(iter) {   
        if(lazy[i] != 0 && left != right) {
			int middle = midPoint;
            apply(lp, lazy[i]), apply(rp, lazy[i]);
            lazy[i] = 0;
        }
    }

	T queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	T queries_at(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    T queries_range(int start, int end) { 
        return queries_range(entireTree, start, end);
    }
    
    T queries_range(iter, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return func(queries_range(lp, start, end), queries_range(rp, start, end));
    }

	void update_window(int L, int R, int len, T x) { // update [l, l + k - 1], [l + 1, l + k], ... [r, r + k] each with x
        update_range(L, L + len - 1, x);
        update_range(R + 1, R + len, -x);
    }

	
	T get() {
		return root[0];
	}
	
	template<typename Pred> // seg.min_left(ending, [](const int& a) {return a > 0;});
        int min_left(int ending, Pred f) { // min index where f[l, ending] is true
            T a = DEFAULT;
            auto ans = find_left(entireTree, ending, f, a);
            return ans == -1 ? ending + 1 : ans;
        }

    template<typename Pred>
        int max_right(int starting, Pred f) {
            T a = DEFAULT;
            auto ans = find_right(entireTree, starting, f, a);
            return ans == -1 ? starting - 1 : ans;
        }

    template<typename Pred>
        int find_left(iter, int end, Pred f, T& now) {
            pushDown;
            if(left > end) return -2;
            if(right <= end && f(func(root[i], now))) {
                now = func(root[i], now);
                return left;
            }
            if(left == right) return -1;
            int middle = midPoint;
            int r = find_left(rp, end, f, now);
            if(r == -2) return find_left(lp, end, f, now);
            if(r == middle + 1) {
                int l = find_left(lp, end, f, now);
                if(l != -1) return l;
            }
            return r;
        }

    template<typename Pred>
        int find_right(iter, int start, Pred f, T &now) {
            pushDown;
            if(right < start) return -2;
            if(left >= start && f(func(now, root[i]))) {
                now = func(now, root[i]);
                return right;
            }
            if(left == right) return -1;
            int middle = midPoint;
            int l = find_right(lp, start, f, now);
            if(l == -2) return find_right(rp, start, f, now);
            if(l == middle) {
                int r = find_right(rp, start, f, now);
                if(r != -1) return r;
            }
            return l;
        }
};

template<class T, typename F = function<T(const T&, const T&)>>
class basic_segtree {
public:
    int n;    
    int size;  
    vt<T> root;
    F func;
    T DEFAULT;  
    
    basic_segtree() {}

    basic_segtree(int n, T DEFAULT, F func = [](const T& a, const T& b) {return a + b;}) : n(n), DEFAULT(DEFAULT), func(func) {
        size = 1;
        while (size < n) size <<= 1;
        root.assign(size << 1, DEFAULT);
    }
    
    void update_at(int idx, T val) {
        if(idx < 0 || idx >= n) return;
        idx += size, root[idx] = val;
        for(idx >>= 1; idx > 0; idx >>= 1) root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
    }
    
    T queries_range(int l, int r) {
        l = max(0, l), r = min(r, n - 1);
        T res_left = DEFAULT, res_right = DEFAULT;
        l += size, r += size;
        while(l <= r) {
            if((l & 1) == 1) res_left = func(res_left, root[l++]);
            if((r & 1) == 0) res_right = func(root[r--], res_right);
            l >>= 1; r >>= 1;
        }
        return func(res_left, res_right);
    }
	
	T queries_at(int idx) {
        if(idx < 0 || idx >= n) return DEFAULT;
        return root[idx + size];
    }

	
	void update_range(int l, int r, ll v) {}

    T get() {
        return root[1];
    }

    template<typename Pred>
    int max_right(int start, Pred P) const {
        if(start < 0) start = 0;
        if(start >= n) return n;
        T sm = DEFAULT;
        int idx = start + size;
        do {
            while((idx & 1) == 0) idx >>= 1;
            if(!P(func(sm, root[idx]))) {
                while(idx < size) {
                    idx <<= 1;
                    T cand = func(sm, root[idx]);
                    if(P(cand)) {
                        sm = cand;
                        idx++;
                    }
                }
                return idx - size - 1;
            }
            sm = func(sm, root[idx]);
            idx++;
        } while((idx & -idx) != idx);
        return n - 1;
    }

    template<typename Pred>
    int min_left(int ending, Pred P) const {
        if(ending < 0) return 0;
        if(ending >= n) ending = n - 1;
        T sm = DEFAULT;
        int idx = ending + size + 1;
        do {
            idx--;
            while(idx > 1 && (idx & 1)) idx >>= 1;
            if(!P(func(root[idx], sm))) {
                while(idx < size) {
                    idx = idx * 2 + 1;
                    T cand = func(root[idx], sm);
                    if(P(cand)) {
                        sm = cand;
                        idx--;
                    }
                }
                return idx + 1 - size;
            }
            sm = func(root[idx], sm);
        } while((idx & -idx) != idx);
        return 0;
    }
};

template<typename T, typename F = function<T(const T, const T)>>
class arithmetic_segtree { // add a + d * (i - left) to [left, right] 
    public: 
    int n;  
    vt<T> root;
    vpll lazy;
    T DEFAULT;
    F func;
    bool is_prefix, inclusive;
	arithmetic_segtree(int n, T DEFAULT, F func = [](const T a, const T b) {return a + b;}, bool is_prefix = true, bool inclusive = true) : n(n), DEFAULT(DEFAULT), is_prefix(is_prefix), inclusive(inclusive), func(func) {    
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1);    
        lazy.rsz(k << 1); 
    }
    
    void update_at(int id, T val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, T val) {  
        pushDown;
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = func(root[lc], root[rc]);
    }

    void update_range(int start, int end, pll val) { 
        update_range(entireTree, start, end, val);
    }
    
    void update_range(iter, int start, int end, pll val) {    
        pushDown;
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
			apply(i, left, right, MP(val.ss * (ll)(is_prefix ? left - start : end - right) + val.ff, val.ss));
			// apply(curr, left, right, {val.ss * (is_prefix ? (left - start) : (end - left)) + val.ff, is_prefix ? val.ss : -val.ss});
            pushDown;
            return;
        }
        int middle = midPoint;  
        update_range(lp, start, end, val);    
        update_range(rp, start, end, val);    
        root[i] = func(root[lc], root[rc]);
    }

	T queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	T queries_at(iter, int id) {
        pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    T queries_range(int start, int end) { 
        return queries_range(entireTree, start, end);
    }
    
    T queries_range(iter, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return func(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	T get() {
		return root[0];
	}
	
	void print() {  
        print(entireTree);
        cout << endl;
    }

    void apply(iter, pll v) {
        ll len = right - left + 1;
        root[i] += len * v.ff + (inclusive ? len * (len + 1) / 2 : len * (len - 1) / 2) * v.ss;
        lazy[i].ff += v.ff;
        lazy[i].ss += v.ss;
    }

    void push(iter) {
        pll zero = MP(0, 0);
        if(lazy[i] != zero && left != right) {
            int middle = midPoint;
            if(is_prefix) {
                apply(lp, lazy[i]);
                pll right_lazy = lazy[i];
                right_lazy.ff += lazy[i].ss * (ll)(middle - left + 1);
                apply(rp, right_lazy);
            } else {
                int middle = midPoint;
                apply(rp, lazy[i]);
                pll left_lazy = lazy[i];
                left_lazy.ff += lazy[i].ss * (ll)(right - middle);
                apply(lp, left_lazy);
            }
            lazy[i] = zero;
        }
    }
};

template<typename T>
struct merge_sort_tree {
    int n;
    vvi arr;
    vt<T> root;
    int res = inf;
    merge_sort_tree(const vi& a) : n(a.size()) {
        int k = 1;
        while(k < n) k <<= 1;
        arr.rsz(k * 2);
        root.rsz(k * 2);
        build(entireTree, a);
    }

    void build(iter, const vi& a) {
        root[i] = inf;
        for(int j = left; j <= right; j++) arr[i].pb(a[j]);
        srt(arr[i]);
        if(left == right) return;
        int middle = midPoint;
        build(lp, a);
        build(rp, a);
    }

    void update_range(int start, int end, int x) {
        update_range(entireTree, start, end, x);
    }

    void update_range(iter, int s, int e, int x) {
        if(left > e || s > right) return;
        if(s <= left && right <= e) {
            auto it = lb(all(arr[i]), x);
            int t = inf;
            if(it != end(arr[i])) t = min(t, abs(*it - x));
            if(it != begin(arr[i])) t = min(t, abs(*--it - x));
            root[i] = min(root[i], t);
            if(t >= res) return;
        }
        if(left == right) {
            res = min(res, root[i]);
            return;
        }
        int middle = midPoint;
        update_range(rp, s, e, x);
        res = min(res, root[rc]);
        update_range(lp, s, e, x);
        root[i] = min(root[lc], root[rc]);
        res = min(res, root[i]);
    }

    int queries_range(int left, int right) {
        return queries_range(entireTree, left, right);
    }

    int queries_range(iter, int s, int e) {
        if(left > e || s > right) return inf;
        if(s <= left && right <= e) return root[i];
        int middle = midPoint;
        return min(queries_range(lp, s, e), queries_range(rp, s, e));
    }
};

class bad_subarray_segtree { 
    // nlog^2n run time
    // for each r, how many l is bad
    // https://codeforces.com/contest/1736/problem/C2
    struct info {
        ll bad;
        int mn, mx;
        info(int x = 0) : bad(x), mn(x), mx(x) {}
    };
    public: 
    int n;  
    vt<info> root;
	bad_subarray_segtree(int n) {    
        this->n = n;
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1, info());    
    }
    
    void update_at(int id, int val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, int val) {  
        if(left == right) { 
            root[i] = info(val);
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = merge(i, left, right);
    }

    ll query_right(iter, int threshold) {
        if(root[i].mn >= threshold) return root[i].bad;
        if(root[i].mx <= threshold) return ((ll)right - left + 1) * threshold;
        int middle = midPoint;
        if(root[lc].mx > threshold) return query_right(lp, threshold) + root[i].bad - root[lc].bad; // the right part got global update by the root[lc].mx already so no need to call it
        return query_right(lp, threshold) + query_right(rp, threshold);
    }

    info merge(iter) {
        int middle = midPoint;
        info res;
        res.mn = min(root[lc].mn, root[rc].mn);
        res.mx = max(root[lc].mx, root[rc].mx);
        res.bad = root[lc].bad + query_right(rp, root[lc].mx);
        return res;
    }

    ll bad_subarray() {
        return root[0].bad; // answer for good subarray is n * (n + 1) / 2 - root[0].bad;
    }
};

// PERSISTENT SEGTREE
int t[MX * MK], ptr, root[MX * 100]; // log2 = MX * 200; careful to match root with the type of template below
pii child[MX * 100]; // maybe * 120
template<class T>
struct PSGT {
    int n;
    T DEFAULT;
    void assign(int n, T DEFAULT) {
        this->DEFAULT = DEFAULT;
        this->n = n;
    }

	void update(int &curr, int prev, int id, T delta, int left, int right) {  
//        if(!curr) curr = ++ptr; // 2d seg to save space
        root[curr] = root[prev];    
        child[curr] = child[prev];
        if(left == right) { 
			root[curr] = merge(root[curr], delta);
            return;
        }
        int middle = midPoint;
        if(id <= middle) child[curr].ff = ++ptr, update(child[curr].ff, child[prev].ff, id, delta, left, middle); // PSGT
        else child[curr].ss = ++ptr, update(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
//        if(id <= middle) update(child[curr].ff, child[prev].ff, id, delta, left, middle); // 2d seg
//        else update(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

	T queries_at(int curr, int start, int end, int left, int right) { 
        if(!curr || left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[curr];
        int middle = midPoint;  
		return merge(queries_at(child[curr].ff, start, end, left, middle), queries_at(child[curr].ss, start, end, middle + 1, right));
    };
        
    T get(int curr, int prev, int k, int left, int right) {    
        if(root[curr] - root[prev] < k) return DEFAULT;
        if(left == right) return left;
        int leftCount = root[child[curr].ff] - root[child[prev].ff];
        int middle = midPoint;
        if(leftCount >= k) return get(child[curr].ff, child[prev].ff, k, left, middle);
        return get(child[curr].ss, child[prev].ss, k - leftCount, middle + 1, right);
    }

    T get(int l, int r, int k) {
        return get(t[r], t[l - 1], k, 0, n - 1);
    }
	
	int find_k(int i, int k) {
        return find_k(t[i], k, 0, n - 1);
    }

    int find_k(int curr, int k, int left, int right) {
        if(root[curr] < k) return inf;
        if(left == right) return left;
        int middle = midPoint;
        if(root[child[curr].ff] >= k) return find_k(child[curr].ff, k, left, middle);
        return find_k(child[curr].ss, k - root[child[curr].ff], middle + 1, right);
    }

    void reset() {  
        for(int i = 0; i <= ptr; i++) { 
            root[i] = 0;
            child[i] = {0, 0};
        }
		for(int i = 0; i < (int)(sizeof(t)/sizeof(t[0])); i++){
            t[i] = 0;
        }
        ptr = 0;
    }

    void add(int i, int& prev, int id, T delta) { 
        t[i] = ++ptr;
        update(t[i], prev, id, delta, 0, n - 1); 
        prev = t[i];
//        while(i < n) { 
//            update(t[i], t[i], id, delta, 0, n - 1);
//            i |= (i + 1);
//        }
    }

    T queries_at(int i, int start, int end) {
        return queries_at(t[i], start, end, 0, n - 1);
//        while(i >= 0) {
//            res += queries(t[i], start, end, 0, n - 1);
//            i = (i & (i + 1)) - 1;
//        }
    }

	T queries_range(int l, int r, int low, int high) {
        if(l > r || low > high) return DEFAULT;
        auto L = (l == 0 ? DEFAULT : queries_at(l - 1, low, high));
        auto R = queries_at(r, low, high);
        return R - L;
    }

    T merge(T left, T right) {
    }
};

struct wavelet_psgt {
    private:
    struct Node {
        int cnt;
        ll sm;
        Node(int cnt = 0, ll sm = 0) : cnt(cnt), sm(sm) {}
    };
    Node merge(const Node& a, const Node& b) {
        return {a.cnt + b.cnt, a.sm + b.sm};
    }
    Node subtract(const Node& a, const Node& b) {
        return {a.cnt - b.cnt, a.sm - b.sm};
    }
    int n;
    vt<Node> root;
    vi t;
    vpii child;
    vi a;
    int new_node() { root.pb(Node(0, 0)); child.pb({0, 0}); return root.size() - 1; }
    int get_id(ll x) { return int(ub(all(a), x) - begin(a)) - 1; }
    public:
    wavelet_psgt() {}

    wavelet_psgt(const vi& arr) : a(arr) {
        t.rsz(arr.size());
        new_node(); 
        srtU(a);
        n = a.size();
        for(int i = 0, prev = 0; i < (int)arr.size(); i++) {
            t[i] = new_node();
            update(t[i], prev, get_id(arr[i]), Node(1, arr[i]), 0, n - 1);
            prev = t[i];
        }
    }

    void update(int curr, int prev, int id, Node delta, int left, int right) {  
        root[curr] = root[prev];    
        child[curr] = child[prev];
        if(left == right) { 
            root[curr] = merge(root[curr], delta);
            return;
        }
        int middle = midPoint;
        if(id <= middle) child[curr].ff = new_node(), update(child[curr].ff, child[prev].ff, id, delta, left, middle); 
        else child[curr].ss = new_node(), update(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

    int kth(int l, int r, int k) {
        return kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    ll sum_kth(int l, int r, int k) {
        return sum_kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    int kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) return -inf;
        if(left == right) return a[left];
        int middle = midPoint;
        int left_cnt = root[child[r].ff].cnt - root[child[l].ff].cnt;
        if(left_cnt >= k) return kth(child[l].ff, child[r].ff, k, left, middle);
        return kth(child[l].ss, child[r].ss, k - left_cnt, middle + 1, right);
    }

    ll sum_kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) return -inf;
        if(k <= 0) return 0;
        if(left == right) return (ll)k * a[left];
        int middle = midPoint;
        int left_cnt = root[child[r].ff].cnt - root[child[l].ff].cnt;
        if(left_cnt >= k) return sum_kth(child[l].ff, child[r].ff, k, left, middle); 
        return root[child[r].ff].sm - root[child[l].ff].sm + sum_kth(child[l].ss, child[r].ss, k - left_cnt, middle + 1, right);
    }

    int median(int l, int r) {
        return kth(l, r, (r - l + 2) / 2);
    }

    Node query_leq(int l, int r, int x) {
        return query((l == 0 ? 0 : t[l - 1]), t[r], 0, get_id(x), 0, n - 1);
    }

    Node query_eq(int l, int r, int x) {
        return subtract(query_leq(l, r, x), query_leq(l, r, x - 1));
    }

    Node queries_range(int l, int r, int low, int high) {
        return query((l == 0 ? 0 : t[l - 1]), t[r], get_id(low - 1) + 1, get_id(high), 0, n - 1);
    }

    Node query(int l, int r, int start, int end, int left, int right) {
        if(left > end || right < start || left > right) return Node();
        if(start <= left && right <= end) return subtract(root[r], root[l]);
        int middle = midPoint;
        return merge(query(child[l].ff, child[r].ff, start, end, left, middle), query(child[l].ss, child[r].ss, start, end, middle + 1, right));
    }
	
	ll first_missing_number(int l, int r) { // https://cses.fi/problemset/task/2184/
        ll s = 1;
        return first_missing_number(l == 0 ? 0 : t[l - 1], t[r], 0, n - 1, s);
    }

    ll first_missing_number(ll l, ll r, ll left, ll right, ll &s) {
        if(s < a[left]) return s;
        Node seg = subtract(root[r], root[l]);
        if(a[right] <= s) {
            s += seg.sm;
            return s;
        }
        ll middle = midPoint;
        first_missing_number(child[l].ff, child[r].ff, left, middle, s);
        first_missing_number(child[l].ss, child[r].ss, middle + 1, right, s);
        return s;
    }
};

template<class T>
struct PSGT {
    struct Node {
        int l, r;
        T key;
        Node(T key) : key(key), l(0), r(0) {}
    };
    int new_node(int prev) {
        F.pb(F[prev]);
        return F.size() - 1;
    }

    int new_node() {
        F.pb(0);
        return F.size() - 1;
    }
    vt<Node> F;
    vi t;
    int n;
    T DEFAULT;
    PSGT(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT), t(n) {
        F.reserve(n * 20);
        F.pb(Node(DEFAULT));
    }

	int update(int prev, int id, T delta, int left, int right) {  
        int curr = new_node(prev);
        if(left == right) { 
            F[curr].key = merge(F[curr].key, delta);
            return curr;
        }
        int middle = midPoint;
        if(id <= middle) F[curr].l = update(F[prev].l, id, delta, left, middle);
        else F[curr].r = update(F[prev].r, id, delta, middle + 1, right);
        F[curr].key = merge(F[F[curr].l].key, F[F[curr].r].key);
        return curr;
    }

	T queries_at(int curr, int start, int end, int left, int right) { 
        if(!curr || left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return F[curr].key;
        int middle = midPoint;  
		return merge(queries_at(F[curr].l, start, end, left, middle), queries_at(F[curr].r, start, end, middle + 1, right));
    };
        
    T get(int curr, int prev, int k, int left, int right) {    
        if(F[curr].key - F[prev].key < k) return DEFAULT;
        if(left == right) return left;
        int leftCount = F[F[curr].l].key - F[F[prev].r].key;
        int middle = midPoint;
        if(leftCount >= k) return get(F[curr].l, F[prev].l, k, left, middle);
        return get(F[curr].r, F[prev].r, k - leftCount, middle + 1, right);
    }

    T get(int l, int r, int k) {
        return get(t[r], t[l - 1], k, 0, n - 1);
    }
	
	int find_k(int i, int k) {
        return find_k(t[i], k, 0, n - 1);
    }

    int find_k(int curr, int k, int left, int right) {
        if(F[curr].key < k) return inf;
        if(left == right) return left;
        int middle = midPoint;
        if(F[F[curr].f].key >= k) return find_k(F[curr].f, k, left, middle);
        return find_k(F[curr].ss, k - F[F[curr].l].key, middle + 1, right);
    }

    void update_at(int i, int& prev, int id, T delta) { 
        t[i] = update(prev, id, delta, 0, n - 1); 
        prev = t[i];
    }

    T queries_at(int i, int start, int end) {
        return queries_at(t[i], start, end, 0, n - 1);
    }

	T queries_range(int l, int r, int low, int high) {
        if(l > r || low > high) return DEFAULT;
        auto L = (l == 0 ? DEFAULT : queries_at(l - 1, low, high));
        auto R = queries_at(r, low, high);
        return R - L;
    }

    T merge(T left, T right) {
        return left + right;
    }
};

struct mex_tree {
    // change merge to min(left, right)
    // change the update to be root[curr] = delta;
    PSGT<int> seg;
    int n;

    mex_tree(const vi& a, int max_value, int starting_mex = 0) : n(max_value), seg(max_value, inf) {
        int prev = 0;
        seg.update_at(0, prev, 0, starting_mex == 0 ? -1 : inf);
        for(int i = 1; i < n; i++) {
            seg.update_at(0, prev, i, -1);
        }
        for (int i = 0; i < (int)a.size(); ++i) {
            int v = min(a[i], n - 1);
            seg.update_at(i + 1, prev, v, i);
        }
    }

    int mex(int l, int r, int k = 1) { // find_kth_mex
        return find_mex(seg.t[r + 1], 0, n - 1, l, k);
    }

    int mex_descending(int l, int r, int lim, int k = 1) {
        return mex_descending(seg.t[r + 1], 0, n - 1, l, lim, k);
    }

private:
    int find_mex(int curr, int L, int R, int bound, int& k) {
        if (L == R) {
            if(--k == 0) return L;
            return -1;
        }
        int M = (L + R) >> 1;
        const auto& F = seg.F;
        if(F[F[curr].l].key < bound) {
            int t = find_mex(F[curr].l, L, M, bound, k);
            if(t != -1) return t;
        }
        if(F[F[curr].r].key < bound) {
            int t = find_mex(F[curr].r, M + 1, R, bound, k);
            if(t != -1) return t;
        }
        return -1;
    }

    int mex_descending(int curr, int L, int R, int bound, int lim, int& k) {
        if(L > lim) return -1;
        if (L == R) {
            if(--k == 0) return L;
            return -1;
        }
        int M = (L + R) >> 1;
        const auto& F = seg.F;
        if(F[F[curr].r].key < bound) {
            int t = mex_descending(F[curr].r, M + 1, R, bound, lim, k);
            if(t != -1) return t;
        }
        if(F[F[curr].l].key < bound) {
            int t = mex_descending(F[curr].l, L, M, bound, lim, k);
            if(t != -1) return t;
        }
        return -1;
    }
};

struct distinct_tree { // range distinct element online
    // modify merging to left + right;
    PSGT<int> root;
    distinct_tree(const vi& a) : root(a.size(), 0) {
        int n = a.size();
        map<int, int> last;
        for(int i = 0, prev = 0; i < n; i++) {
            int x = a[i];
            if(last.count(x)) {
                root.update_at(i, prev, last[x], -1);
            } 
            root.update_at(i, prev, i, 1);
            last[x] = i;
        }
    }  

    int query(int l, int r) {
        return root.queries_at(r, l, r);
    }
};

struct good_split {
    // determine if in [l, r], there's an index such that max([l, i]) < min([i + 1, r])
    vi a;
    int n;
    PSGT<int> Tree;
    // merge is min, and root[curr] = delta
    good_split(const vi& a) : n(a.size()), a(a), Tree(n, inf) {
        // https://codeforces.com/contest/1887/problem/D
        auto L = closest_left(a, less<int>());
        linear_rmq<int> t(a, [](const int& x, const int& y) {return x > y;});
        int prev = 0;
        for(int i = 0; i < n; i++) {
            Tree.update_at(0, prev, i, inf);
        }
        set<int> s;
        for(int r = 1; r < n; r++) {
            for(auto it = s.lb(L[r]); it != end(s);) {
                Tree.update_at(r, prev, *it, inf);
                it = s.erase(it);
            }
            int left = 0, right = L[r] - 1, right_most = 0;
            while(left <= right) {
                int middle = midPoint;
                if(t.query(middle, L[r] - 1) > a[r]) right_most = middle, left = middle + 1;
                else right = middle - 1;
            }
            if(L[r] - 1 > 0) {
                Tree.update_at(r, prev, L[r] - 1, right_most);
                s.insert(L[r] - 1);
            }
        }
    }

    int query(int l, int r) {
        return Tree.queries_at(r, l, r) < l;
    }
};

struct LCM_tree {
    // do merge as left * right
    // careful with the memory, memory should be MX * 240
    // MX initializer should be meeting the constraint
    // have an init variable in the psgt to init everything with 1
    // do the DIV as vpii for [prime, cnt]
    // https://codeforces.com/contest/1422/problem/F
    PSGT<mint> Tree;
    LCM_tree(const vi& a) {
        Tree.reset();
        int n = a.size();
        Tree.assign(n, 1);
        int prev = 0;
        for(int i = 0; i < n; i++) {
            Tree.add(0, prev, i, 1);
        }
        Tree.init = false;
        const int N = MAX(a);
        stack<pii> s[N + 1];
        for(int i = 1; i <= n; i++) {
            t[i] = prev;
            int X = a[i - 1];
            for(auto& [x, cnt] : DIV[X]) {
                auto& curr = s[x];
                int last = 0;
                while(!curr.empty() && curr.top().ss <= cnt) {
                    auto [j, c] = curr.top(); curr.pop();
                    assert(c >= last);
                    Tree.add(i, prev, j, mint(1) / mint(x).pow(c - last));
                    last = c;
                }
                auto now = mint(x).pow(cnt);
                if(!curr.empty() && cnt > last) {
                    auto [j, oldCnt] = curr.top();
                    Tree.add(i, prev, j, mint(1) / mint(x).pow(cnt - last));
                }
                Tree.add(i, prev, i - 1, now);
                curr.push({i - 1, cnt});
            } 
        }
    } 

    mint query(int l, int r) {
        return Tree.queries_at(r + 1, l, r);
    }
};

// you have to set up by assigning size and updating from 0 to n - 1 first
template<typename T>
struct lazy_PSGT {
    struct Node {
        T s, g;
        ll lazy;
        int l, r;
        Node(T key = 0) : s(key), g(key), lazy(1), l(0), r(0) { }

        friend Node operator+(const Node& a, const Node& b) {
            Node res;
            res.g = gcd(a.g, b.g);
            res.s = a.s + b.s;
            return res;
        }
    };
    vt<Node> F;
    vi t;
    int n;
    T DEFAULT;
    lazy_PSGT(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT), t(n + 10) {
        F.reserve(n * 50);
        F.pb(Node(DEFAULT));
    }

    int new_node(int prev) {
        F.pb(F[prev]);
        return int(F.size()) - 1;
    }

    void pull(int curr) {
        int l = F[curr].l;
        int r = F[curr].r;
        F[curr] = F[l] + F[r];
        F[curr].l = l;
        F[curr].r = r;
    }

    void apply(int curr, int left, int right, T val) {
        auto& x = F[curr];
        x.s /= val;
        x.g /= val;
        x.lazy *= val;
    }

    void push_down(int curr, int left, int right) {
        if(left == right || !curr || F[curr].lazy == 1) return;
        int middle = (left + right) >> 1;
        if(F[curr].l) {
            F[curr].l = new_node(F[curr].l);
            apply(F[curr].l, left, middle, F[curr].lazy);
        }
        if(F[curr].r) {
            F[curr].r = new_node(F[curr].r);
            apply(F[curr].r, middle + 1, right, F[curr].lazy);
        }
        F[curr].lazy = 1;
    }

    int update_range(int prev, int start, int end, T delta) {
        return update_range(prev, start, end, 0, n - 1, delta);
    }

    int update_range(int prev, int start, int end, int left, int right, T delta) {
        if(left > end || start > right) return prev;
        int curr = new_node(prev);
        push_down(curr, left, right);
        if(start <= left && right <= end) {
            apply(curr, left, right, delta);
            push_down(curr, left, right);
            return curr;
        }
        int middle = midPoint;
        F[curr].l = update_range(F[curr].l, start, end, left, middle, delta);
        F[curr].r = update_range(F[curr].r, start, end, middle + 1, right, delta);
        pull(curr);
        return curr;
    }

    Node queries_range(int i, int start, int end) {
        return queries_range(i, start, end, 0, n - 1);
    }

    Node queries_range(int curr, int start, int end, int left, int right) {
        push_down(curr, left, right);
        if(!curr || start > right || left > end) return Node();
        if(start <= left && right <= end) return F[curr];
        int middle = midPoint;
        return queries_range(F[curr].l, start, end, left, middle) + queries_range(F[curr].r, start, end, middle + 1, right);
    }

    int update_at(int prev, int id, T delta) {
        return update_at(prev, id, delta, 0, n - 1);
    }

    int update_at(int prev, int id, T delta, int left, int right) {
        int curr = new_node(prev);
        if(left == right) {
            F[curr] = Node(delta);
            return curr;
        }
        int middle = midPoint;
        if(id <= middle) {
            F[curr].l = update_at(F[prev].l, id, delta, left, middle);
        } else {
            F[curr].r = update_at(F[prev].r, id, delta, middle + 1, right);
        }
        pull(curr);
        return curr;
    }
};

template<class T>
struct SGT_2D {
    vt<vt<T>> root;
    int n, m, N;           
    T DEFAULT;             

    SGT_2D(int n, int m, T DEFAULT) {
        this->n = n;
        this->m = m;
        this->DEFAULT = DEFAULT;
        this->N = max(n, m); 
        root.resize(N * 2, vt<T>(N * 2)); // do 4 * N for recursive segtreee
    }

    void update_at(int x, int y, T value) {
        x += N; y += N;
        root[x][y] = value;
        for (int ty = y; ty > 1; ty >>= 1) {
            root[x][ty >> 1] = merge(root[x][ty], root[x][ty ^ 1]);
        }
        for (int tx = x; tx > 1; tx >>= 1) {
            for (int ty = y; ty >= 1; ty >>= 1) {
                root[tx >> 1][ty] = merge(root[tx][ty], root[tx ^ 1][ty]);
            }
        }
    }

    T queries_range(int start_x, int end_x, int start_y, int end_y) {
        start_x += N; end_x += N;    
        start_y += N; end_y += N;   
        T result = DEFAULT;

        while (start_x <= end_x) {
            if (start_x & 1) { 
                int sy = start_y, ey = end_y;
                while (sy <= ey) {
                    if (sy & 1) result = merge(result, root[start_x][sy++]);
                    if (!(ey & 1)) result = merge(result, root[start_x][ey--]);
                    sy >>= 1; ey >>= 1;
                }
                start_x++;
            }
            if (!(end_x & 1)) {
                int sy = start_y, ey = end_y;
                while (sy <= ey) {
                    if (sy & 1) result = merge(result, root[end_x][sy++]);
                    if (!(ey & 1)) result = merge(result, root[end_x][ey--]);
                    sy >>= 1; ey >>= 1;
                }
                end_x--;
            }
            start_x >>= 1;
            end_x >>= 1;
        }
        return result;
    }

	T queries_at(int r, int c) {
        return queries_range(r, r, c, c);
    }

    T merge(T A, T B) {
    }

//    void update_at(int x, int y, T v) {
//        update_at(x, y, v, 0, n - 1, 0, m - 1, 1, 1);
//    }
// 
//    void update_at(int x, int y, T val, int left_x, int right_x, int left_y, int right_y, int node_x, int node_y) {
//        if (left_x == right_x && left_y == right_y) {
//            root[node_x][node_y] = val;
//            return;
//        }
//        int mid_x = (left_x + right_x) / 2;
//        int mid_y = (left_y + right_y) / 2;
//        if (x <= mid_x && y <= mid_y) update_at(x, y, val, left_x, mid_x, left_y, mid_y, 2 * node_x, 2 * node_y); 
//        else if (x <= mid_x) update_at(x, y, val, left_x, mid_x, mid_y + 1, right_y, 2 * node_x, 2 * node_y + 1);
//        else if (y <= mid_y) update_at(x, y, val, mid_x + 1, right_x, left_y, mid_y, 2 * node_x + 1, 2 * node_y);
//        else update_at(x, y, val, mid_x + 1, right_x, mid_y + 1, right_y, 2 * node_x + 1, 2 * node_y + 1);
//        root[node_x][node_y] = merge(
//            root[2 * node_x][2 * node_y],
//            root[2 * node_x][2 * node_y + 1],
//            root[2 * node_x + 1][2 * node_y],
//            root[2 * node_x + 1][2 * node_y + 1]
//        );
//    }
// 
//    T queries_range(int start_x, int end_x, int start_y, int end_y) {
//        return queries_range(start_x, end_x, start_y, end_y, 0, n - 1, 0, m - 1, 1, 1);
//    }
// 
//    T queries_range(int start_x, int end_x, int start_y, int end_y, int left_x, int right_x, int left_y, int right_y, int node_x, int node_y) {
//        if (start_x > right_x || end_x < left_x || start_y > right_y || end_y < left_y) return DEFAULT;
//        if (start_x <= left_x && right_x <= end_x && start_y <= left_y && right_y <= end_y) return root[node_x][node_y];
//        int mid_x = (left_x + right_x) / 2;
//        int mid_y = (left_y + right_y) / 2;
//        return merge(
//            queries_range(start_x, end_x, start_y, end_y, left_x, mid_x, left_y, mid_y, 2 * node_x, 2 * node_y),
//            queries_range(start_x, end_x, start_y, end_y, left_x, mid_x, mid_y + 1, right_y, 2 * node_x, 2 * node_y + 1),
//            queries_range(start_x, end_x, start_y, end_y, mid_x + 1, right_x, left_y, mid_y, 2 * node_x + 1, 2 * node_y),
//            queries_range(start_x, end_x, start_y, end_y, mid_x + 1, right_x, mid_y + 1, right_y, 2 * node_x + 1, 2 * node_y + 1)
//        );
//    }
// 
//    T merge(T A, T B, T C, T D) {
//        return min(A, min(B, min(C, D)));
//    }
};

int root[MX * 120], lazy[MX * 120], ptr; // MX should be 1e5
pii child[MX * 120];
class implicit_segtree {
    public:
    int n;
    implicit_segtree(int n) {
        this->n = n;
        root[0] = a.queries(0, n - 1); // initialize
        lazy[0] = -1;
    } 

    void create_node(int& node, int left, int right) {
        if(node) return;
        node = ++ptr;
        lazy[node] = -1;
        root[node] = a.queries(left, right);
    }

    void update(int start, int end, int x) {
        update(entireTree, start, end, x);
    }

    void update(iter, int start, int end, int x) {
        pushDown;
        if(left > end || start > right) return;
        if(start <= left && right <= end) {
			apply(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        update(child[i].ff, left, middle, start, end, x);
        update(child[i].ss, middle + 1, right, start, end, x);
        root[i] = merge(root[child[i].ff], root[child[i].ss]);
    }

	void apply(iter, int val) {
        root[i] = val * (right - left + 1);
        lazy[i] = val;
    }

    void push(iter) {
        if(lazy[i] != -1 && left != right) {
            int middle = midPoint;
            create_node(child[i].ff, left, middle);
            create_node(child[i].ss, middle + 1, right);
			apply(child[i].ff, left, middle, lazy[i]);
            apply(child[i].ss, middle + 1, right, lazy[i]);
            lazy[i] = -1;
        }
    }

    int merge(int left, int right) {
        return min(left, right);
    }

    int queries(int start, int end) {
        return queries(entireTree, start, end);
    }

    int queries(iter, int start, int end) {
        pushDown;
        if(start > right || left > end) return inf;
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        return merge(queries(child[i].ff, left, middle, start, end), queries(child[i].ss, middle + 1, right, start, end));
    }
	
	void update(int& i, int x) {
        update(i, 0, n - 1, x);
    }

    void update(int& i, int left, int right, int x) {
        if(!i) i = ++ptr;
        if(left == right) return;
        int middle = midPoint;
        if(x <= middle) update(child[i].ff, left, middle, x);
        else update(child[i].ss, middle + 1, right, x);
    }


    int merge_two_tree(int i, int j) {
        if(!i || !j) return i + j;
        child[i].ff = merge_two_tree(child[i].ff, child[j].ff);
        child[i].ss = merge_two_tree(child[i].ss, child[j].ss);
        return i;
    }

    void modify_two_tree(int& i, int& j, int start, int end) {
        modify(i, j, 0, n - 1, start, end);
    }

    void modify(int& i, int& j, int left, int right, int start, int end) {
        if(!i || left > end || start > right) return;
        if(!j) j = ++ptr;
        if(left >= start && right <= end) {
            j = merge_two_tree(i, j);
            i = 0;
            return;
        }
        int middle = midPoint;
        modify(child[i].ff, child[j].ff, left, middle, start, end);
        modify(child[i].ss, child[j].ss, middle + 1, right, start, end);
    }

};

template<class T>
class SGT_BEAT {
    private:
    struct Node {
        T mx1, mx2, mn1, mn2, mx_cnt, mn_cnt, sm, ladd, lval;
        Node(T x = INF) : mx1(x), mx2(-INF), mn1(x), mn2(INF), mx_cnt(1), mn_cnt(1), sm(x), lval(INF), ladd(0) {}
    };

    Node merge(const Node& left, const Node& right) {
        if(left.mx1 == INF) return right;
        if(right.mx1 == INF) return left;
        Node res;
        res.sm = left.sm + right.sm;
        if(left.mx1 > right.mx1) {
            res.mx1 = left.mx1;
            res.mx_cnt = left.mx_cnt;
            res.mx2 = max(left.mx2, right.mx1);
        } else if(left.mx1 < right.mx1) {
            res.mx1 = right.mx1;
            res.mx_cnt = right.mx_cnt;
            res.mx2 = max(left.mx1, right.mx2);
        } else {
            res.mx1 = left.mx1;
            res.mx_cnt = left.mx_cnt + right.mx_cnt;
            res.mx2 = max(left.mx2, right.mx2);
        }

        if(left.mn1 < right.mn1) {
            res.mn1 = left.mn1;
            res.mn_cnt = left.mn_cnt;
            res.mn2 = min(left.mn2, right.mn1);
        } else if(left.mn1 > right.mn1) {
            res.mn1 = right.mn1;
            res.mn_cnt = right.mn_cnt;
            res.mn2 = min(right.mn2, left.mn1);
        } else {
            res.mn1 = left.mn1;
            res.mn_cnt = left.mn_cnt + right.mn_cnt;
            res.mn2 = min(left.mn2, right.mn2);
        }
        return res;
    }

    void update_at(iter, int id, T x) {
        pushDown;
        if(left == right) {
            root[i] = Node(x);
            return;
        }
        int middle = midPoint;
        if(id <= middle) update_at(lp, id, x);
        else update_at(rp, id, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_min(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end || root[i].mx1 <= x) return;
        if(start <= left && right <= end && root[i].mx2 < x) {
            update_node_max(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_min(lp, start, end, x);
        update_min(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_max(iter, int start, int end, T x) {
        pushDown;
        if(left > end || start > right || x <= root[i].mn1) return;
        if(start <= left && right <= end && x < root[i].mn2) {
			update_node_min(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_max(lp, start, end, x);
        update_max(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
        
    }

    void update_node_min(iter, T x) {
        root[i].sm += (x - root[i].mn1) * root[i].mn_cnt;  
        if(root[i].mn1 == root[i].mx1) {
            root[i].mn1 = root[i].mx1 = x;
        } else if(root[i].mn1 == root[i].mx2) {
            root[i].mn1 = root[i].mx2 = x;
        } else {
            root[i].mn1 = x;
        }
    }

    void update_node_max(iter, T x) {
        root[i].sm += (x - root[i].mx1) * root[i].mx_cnt;
        if(root[i].mx1 == root[i].mn1) {
            root[i].mx1 = root[i].mn1 = x;
        } else if(root[i].mx1 == root[i].mn2) {
            root[i].mx1 = root[i].mn2 = x;
        } else {
            root[i].mx1 = x;
        }
    }

    void update_val(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end) return;
        if(start <= left && right <= end) {
            update_all(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_val(lp, start, end, x);
        update_val(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void update_all(iter, T x) {
        root[i] = Node(x);
        T len = right - left + 1;
        root[i].sm = len * x;
        root[i].mx_cnt = root[i].mn_cnt = len;
        root[i].lval = x;
    }

    void update_add(iter, int start, int end, T x) {
        pushDown;
        if(start > right || left > end) return;
        if(start <= left && right <= end) {
            add_val(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_add(lp, start, end, x);
        update_add(rp, start, end, x);
        root[i] = merge(root[lc], root[rc]);
    }

    void add_val(iter, T x) {
        root[i].mx1 += x;
        if(root[i].mx2 != -INF) root[i].mx2 += x;
        root[i].mn1 += x;
        if(root[i].mn2 != INF) root[i].mn2 += x;
        root[i].sm += x * (right - left + 1);
        if(root[i].lval != INF) root[i].lval += x;
        else root[i].ladd += x;
    }

    void push(iter) {
        if(left == right) return;    
        int middle = midPoint;
        if(root[i].lval != INF) {
            update_all(lp, root[i].lval);
            update_all(rp, root[i].lval);
            root[i].lval = INF;
            return;
        }
        if(root[i].ladd) {
            add_val(lp, root[i].ladd);
            add_val(rp, root[i].ladd);
            root[i].ladd = 0;
        }
        if(root[i].mx1 < root[lc].mx1) update_node_max(lp, root[i].mx1);
        if(root[i].mn1 > root[lc].mn1) update_node_min(lp, root[i].mn1);
        if(root[i].mx1 < root[rc].mx1) update_node_max(rp, root[i].mx1);
        if(root[i].mn1 > root[rc].mn1) update_node_min(rp, root[i].mn1);
    }

    Node queries_range(iter, int start, int end) {
        pushDown;
        if(left > end || start > right) return Node();
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        return merge(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	Node queries_at(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    public:
    int n;
    vt<Node> root;
    SGT_BEAT(int n) {
        this->n = n;
        int k = 1;
        while(k < n) k <<= 1;
        root.rsz(k << 1);
    }

    void update_at(int id, T x) { update_at(entireTree, id, x); }
    void update_min(int start, int end, T x) { update_min(entireTree, start, end, x); }
    void update_max(int start, int end, T x) { update_max(entireTree, start, end, x); }
    void update_val(int start, int end, T x) { update_val(entireTree, start, end, x); }
    void update_add(int start, int end, T x) { update_add(entireTree, start, end, x); }
    Node queries_range(int start, int end) { return queries_range(entireTree, start, end); }
	Node queries_at(int id) { return queries_at(entireTree, id); }
	
    template<typename OP>
    // call by update_unary(l, r, x, [](const int& a, const int& b) {return a % b;});
    void update_unary(int start, int end, T x, OP c) { // update range and, range or, range divide, range mod, ... anything that's unary
        update_unary(entireTree, start, end, x, c);
    }
    
    template<typename OP>
    void update_unary(iter, int start, int end, T x, OP op) {
        pushDown;
        if(start > right || left > end) return; // for range mod do a return if root[i].mx1 < x
        if(start <= left && right <= end && root[i].mx1 == root[i].mn1) {
            T nv = op(root[i].mx1, x);
            update_all(i, left, right, nv);
            pushDown;
            return;
        }
        int middle = midPoint;
        update_unary(lp, start, end, x, op);
        update_unary(rp, start, end, x, op);
        root[i] = merge(root[lc], root[rc]);
    }
};

struct HISTORICAL_SGT_BEAT {
    // having two same array a and b at the start
    // store historical mn and historical mx 
    // meaning it's the lowest a[i] gets to at any point, same for mx in b[i]
    // https://uoj.ac/problem/169
    struct node {
        int mn, hmn, se;
        int mx, hmx, le, hle;
        int tag1, htag1, tag2, htag2, tag3, htag3, tag4, htag4;
        node(ll val = inf)
            : mn(val), hmn(val), se(inf),
              mx(val), hmx(val), le(val), hle(val),
              tag1(0), htag1(0), tag2(0), htag2(0), tag3(0), htag3(0), tag4(0), htag4(0) {}
    };
    vt<node> tree;
    int n;

    HISTORICAL_SGT_BEAT(int _n = 0) : n(_n) {
        int k = 1;
        while(k < n) k <<= 1;
        tree.rsz(k << 1);
    }

    node merge(const node &L, const node &R) {
        if(L.mn == inf) return R;
        if(R.mn == inf) return L;
        node res;
        res.mn = min(L.mn, R.mn);
        res.hmn = min(L.hmn, R.hmn);
        if(L.mn == R.mn) res.se = min(L.se, R.se);
        else if(L.mn < R.mn) res.se = min(L.se, R.mn);
        else res.se = min(L.mn, R.se);
        res.mx = max(L.mx, R.mx);
        res.hmx = max(L.hmx, R.hmx);
        if(L.mx == R.mx) res.le = max(L.le, R.le);
        else if(L.mx > R.mx) res.le = max(L.le, R.mx);
        else res.le = max(L.mx, R.le);
        res.hle = max(L.hle, R.hle);
        res.tag1 = res.htag1 = res.tag2 = res.htag2 = 0;
        res.tag3 = res.htag3 = res.tag4 = res.htag4 = 0;
        return res;
    }

    void push_up(int i) {
        tree[i].mn = min(tree[lc].mn,  tree[rc].mn);
        tree[i].hmn = min(tree[lc].hmn, tree[rc].hmn);
        if(tree[lc].mn == tree[rc].mn) tree[i].se = min(tree[lc].se, tree[rc].se);
        else if(tree[lc].mn < tree[rc].mn) tree[i].se = min(tree[lc].se, tree[rc].mn);
        else tree[i].se = min(tree[lc].mn, tree[rc].se);
        tree[i].mx = max(tree[lc].mx,  tree[rc].mx);
        tree[i].hmx = max(tree[lc].hmx, tree[rc].hmx);
        if(tree[lc].mx == tree[rc].mx) tree[i].le = max(tree[lc].le, tree[rc].le);
        else if (tree[lc].mx > tree[rc].mx) tree[i].le = max(tree[lc].le, tree[rc].mx);
        else tree[i].le = max(tree[lc].mx, tree[rc].le);
        tree[i].hle = max(tree[lc].hle, tree[rc].hle);
    }

    void push_tag1(int i, int tag, int htag) {
        tree[i].hmn = min(tree[i].hmn, tree[i].mn + htag);
        tree[i].mn += tag;
        tree[i].htag1 = min(tree[i].htag1, tree[i].tag1 + htag);
        tree[i].tag1 += tag;
    }

    void push_tag2(int i, int tag, int htag) {
        if(tree[i].se != inf) tree[i].se += tag;
        tree[i].htag2 = min(tree[i].htag2, tree[i].tag2 + htag);
        tree[i].tag2 += tag;
    }

    void push_tag3(int i, int tag, int htag) {
        tree[i].hmx = max(tree[i].hmx, tree[i].mx + htag);
        tree[i].mx += tag;
        tree[i].htag3 = max(tree[i].htag3, tree[i].tag3 + htag);
        tree[i].tag3 += tag;
    }

    void push_tag4(int i, int tag, int htag) {
        tree[i].hle = max(tree[i].hle, tree[i].le + htag);
        tree[i].le += tag;
        tree[i].htag4 = max(tree[i].htag4, tree[i].tag4 + htag);
        tree[i].tag4 += tag;
    }

    void push(iter) {
        if(left == right) return;
        int middle = midPoint;
        int mv = min(tree[lc].mn, tree[rc].mn);
        if(tree[lc].mn <= mv) push_tag1(lc, tree[i].tag1, tree[i].htag1);
        else push_tag1(lc, tree[i].tag2, tree[i].htag2);
        push_tag2(lc, tree[i].tag2, tree[i].htag2);
        push_tag3(lc, tree[i].tag3, tree[i].htag3);
        push_tag4(lc, tree[i].tag4, tree[i].htag4);
        if(tree[rc].mn <= mv) push_tag1(rc, tree[i].tag1, tree[i].htag1);
        else push_tag1(rc, tree[i].tag2, tree[i].htag2);
        push_tag2(rc, tree[i].tag2, tree[i].htag2);
        push_tag3(rc, tree[i].tag3, tree[i].htag3);
        push_tag4(rc, tree[i].tag4, tree[i].htag4);
        tree[i].tag1 = tree[i].htag1 = tree[i].tag2 = tree[i].htag2 = 0;
        tree[i].tag3 = tree[i].htag3 = tree[i].tag4 = tree[i].htag4 = 0;
    }

    void update_add(iter, int l, int r, int k) {
        if(l <= left && right <= r) {
            push_tag1(i, k, k);
            push_tag2(i, k, k);
            push_tag3(i, k, k);
            push_tag4(i, k, k);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(l <= middle) update_add(lp, l, r, k);
        if(r > middle)  update_add(rp, l, r, k);
        push_up(i);
    }

    void update_max(iter, int l, int r, int k) {
        if(tree[i].mn >= k) return;
        if(l <= left && right <= r && tree[i].se > k) {
            int delta = k - tree[i].mn;
            push_tag1(i, delta, delta);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(l <= middle) update_max(lp, l, r, k);
        if(r > middle) update_max(rp, l, r, k);
        push_up(i);
    }

    void update_at(iter, int pos, int val) {
        if(left == right) {
            tree[i] = node(val);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(pos <= middle) update_at(lp, pos, val);
        else update_at(rp, pos, val);
        push_up(i);
    }

    void update_min(iter, int l, int r, int x) {
        if(tree[i].mx <= x) return;
        if(l <= left && right <= r && tree[i].le < x) {
            int delta = x - tree[i].mx;
            push_tag3(i, delta, delta);
            return;
        }
        pushDown;
        int middle = midPoint;
        if(l <= middle) update_min(lp, l, r, x);
        if(r > middle) update_min(rp, l, r, x);
        push_up(i);
    }

    node queries_at(iter, int pos) {
        if(left == right) return tree[i];
        pushDown;
        int middle = midPoint;
        return pos <= middle ? queries_at(lp, pos)
                              : queries_at(rp, pos);
    }

    node queries_range(iter, int l, int r) {
        if (l <= left && right <= r) return tree[i];
        pushDown;
        int middle = midPoint;
        if (r <= middle) return queries_range(lc, left, middle, l, r);
        if (l > middle)  return queries_range(rc, middle + 1, right, l, r);
        node L = queries_range(lc, left, middle, l, r);
        node R = queries_range(rc, middle + 1, right, l, r);
        return merge(L, R);
    }

    void update_add(int l, int r, int x) { update_add(entireTree, l, r, x); }
    void update_max(int l, int r, int x) { update_max(entireTree, l, r, x); }
    void update_min(int l, int r, int x) { update_min(entireTree, l, r, x);};
    void update_at(int pos, int x) { update_at(entireTree, pos, x); }
    node queries_at(int pos) { return queries_at(entireTree, pos); }
    node queries_range(int l, int r) { return queries_range(entireTree, l, r); }
};

struct hash_info {
    int len;
    ll fwd[2], rev[2];

    hash_info(ll v = -1) : len(v != -1) {
        fwd[0] = fwd[1] = rev[0] = rev[1] = v;
    }

    bool is_palindrome() const {
        return len > 0 && fwd[0] == rev[0] && fwd[1] == rev[1];
    }

    friend hash_info operator+(const hash_info& a, const hash_info& b) {
        if (a.len == 0) return b;
        if (b.len == 0) return a;
        hash_info r;
        r.len = a.len + b.len;
        for(int i = 0; i < 2;i++){
            r.fwd[i] = (a.fwd[i] * p[i][b.len] + b.fwd[i]) % mod[i];
            r.rev[i] = (b.rev[i] * p[i][a.len] + a.rev[i]) % mod[i];
        }
        return r;
    }
};

struct max_subarray_info {
    ll ans, prefix, suffix, sm;
    max_subarray_info(ll x = -INF) : ans(max(0LL, x)), prefix(max(0LL, x)), suffix(max(0LL, x)), sm(x) {}

    friend max_subarray_info operator+(const max_subarray_info& a, const max_subarray_info& b) {
        if(a.sm == -INF) return b;
        if(b.sm == -INF) return a;
        max_subarray_info res;
        res.ans = max({a.ans, b.ans, a.suffix + b.prefix});
        res.sm = a.sm + b.sm;
        res.prefix = max(a.prefix, a.sm + b.prefix);
        res.suffix = max(b.suffix, b.sm + a.suffix);
        return res;
    }
};

struct info_0_1_0 { // maximum of segment of left0 + right0 where mid is a one
    // https://leetcode.com/problems/maximize-active-section-with-trade-ii/submissions/1590475876/
    int one_left, one_right, zero_left, zero_right,
        ans, sm, zero_left2, zero_right2, one_left2, one_right2;
    info_0_1_0(int x = -1)
        : one_left(x == 1),
          one_right(x == 1),
          zero_left(x == 0),
          zero_right(x == 0),
          ans(0),
          sm(x != -1),
          zero_left2(x == 0),
          zero_right2(x == 0),
          one_left2(x == 1),
          one_right2(x == 1)
    {}

    friend info_0_1_0 operator+(const info_0_1_0& a, const info_0_1_0& b) {
        if(a.sm == 0) return b;
        if(b.sm == 0) return a;
        info_0_1_0 res;
        res.sm = a.sm + b.sm;
        res.ans = max(a.ans, b.ans);
        res.ans = max(a.ans, b.ans);
        if(b.one_left && a.one_right && a.zero_right2 && b.zero_left2) res.ans = max(res.ans, a.zero_right2 + b.zero_left2);
        else if(a.one_right && a.zero_right2 && b.zero_left) res.ans = max(res.ans, a.zero_right2 + b.zero_left);
        else if(b.one_left && a.zero_right && b.zero_left2) res.ans = max(res.ans, a.zero_right + b.zero_left2);
        else {
            if(a.one_right2 && a.zero_right2 && b.zero_left) {
                res.ans = max(res.ans, a.zero_right + b.zero_left + a.zero_right2);
            }
            if(b.one_left2 && b.zero_left2 && a.zero_right) {
                res.ans = max(res.ans, a.zero_right + b.zero_left + b.zero_left2);
            }
        }
        if(a.one_left) {
            res.one_left = a.one_left;
            if(a.one_left == a.sm) {
                if(b.one_left) {
                    res.one_left += b.one_left;
                    res.zero_left2 = b.zero_left2;
                } else {
                    res.zero_left2 = b.zero_left;
                }
                res.one_left2 = b.one_left2;
            } else {
                res.zero_left2 = a.zero_left2;
                if(a.one_left + a.zero_left2 == a.sm) {
                    if(b.zero_left) {
                        res.zero_left2 += b.zero_left;
                        res.one_left2 = b.one_left2;
                    } else {
                        res.one_left2 = b.one_left;
                    }
                } else {
                    res.one_left2 = a.one_left2 + (a.one_left + a.zero_left2 + a.one_left2 == a.sm ? b.one_left : 0);
                }
            }
        }
        else {
            res.zero_left = a.zero_left;
            if(a.zero_left == a.sm) {
                if(b.zero_left) {
                    res.zero_left += b.zero_left;
                    res.one_left2 = b.one_left2;
                } else {
                    res.one_left2 = b.one_left;
                }
                res.zero_left2 = b.zero_left2;
            } else {
                res.one_left2 = a.one_left2;
                if(a.zero_left + a.one_left2 == a.sm) {
                    if(b.one_left) {
                        res.one_left2 += b.one_left;
                        res.zero_left2 = b.zero_left2;
                    } else {
                        res.zero_left2 = b.zero_left;
                    }
                } else {
                    res.zero_left2 = a.zero_left2 + (a.zero_left + a.one_left2 + a.zero_left2 == a.sm ? b.zero_left : 0);
                }
            }
        }
        if(b.one_right) {
            res.one_right = b.one_right;
            if(b.one_right == b.sm) {
                if(a.one_right) {
                    res.one_right += a.one_right;
                    res.zero_right2 = a.zero_right2;
                } else {
                    res.zero_right2 = a.zero_right;
                }
                res.one_right2 = a.one_right2;
            } else {
                res.zero_right2 = b.zero_right2;
                if(b.one_right + b.zero_right2 == b.sm) {
                    if(a.zero_right) {
                        res.zero_right2 += a.zero_right;
                        res.one_right2 = a.one_right2;
                    } else {
                        res.one_right2 = a.one_right;
                    }
                } else {
                    res.one_right2 = b.one_right2 + (b.one_right + b.one_right2 + b.zero_right2 == b.sm ? a.one_right : 0);
                }
            }
        } else {
            res.zero_right = b.zero_right;
            if(b.zero_right == b.sm) {
                if(a.zero_right) {
                    res.zero_right += a.zero_right;
                    res.one_right2 = a.one_right2;
                } else {
                    res.one_right2 = a.one_right;
                }
                res.zero_right2 = a.zero_right2;
            } else {
                res.one_right2 = b.one_right2;
                if(b.zero_right + b.one_right2 == b.sm) {
                    if(a.one_right) {
                        res.one_right2 += a.one_right;
                        res.zero_right2 = a.zero_right2;
                    } else {
                        res.zero_right2 = a.zero_right;
                    }
                } else {
                    res.zero_right2 = b.zero_right2 + (b.zero_right + b.one_right2 + b.zero_right2 == b.sm ? a.zero_right : 0);
                }
            }
        }
        return res;
    }
};

struct dp_info { // knapsack pick not pick
    ll dp[2][2];
    dp_info(ll x = -INF) {
        mset(dp, 0);
        if(x > 0) dp[1][1] = x;
    }

    friend dp_info operator+(const dp_info& a, const dp_info& b) {
        dp_info res;
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                res.dp[i][j] = max({res.dp[i][j], 
                                    a.dp[i][0] + b.dp[0][j], 
                                    a.dp[i][1] + b.dp[0][j], 
                                    a.dp[i][0] + b.dp[1][j]});
            }
        }
        return res;
    }
};

struct bracket_info { // balance bracket sequence
    int sm, min_prefix;
    bracket_info(int x = 0) : sm(x), min_prefix(min(x, 0)) {}

    bool is_balance() {
        return sm == 0 && min_prefix >= 0;
    }

    friend bracket_info operator+(const bracket_info& a, const bracket_info& b) {
        bracket_info res;
        res.sm = a.sm + b.sm;
        res.min_prefix = min(a.min_prefix, a.sm + b.min_prefix);
        return res;
    }
};

struct bracket_subsequence_info { // maximum balance bracket [l, r]
    // https://codeforces.com/contest/380/problem/C
    int open, close, ans;
    bracket_subsequence_info(int x = 0) : ans(0), open(x == 1), close(x == -1) {}

    friend bracket_subsequence_info operator+(const bracket_subsequence_info& a, const bracket_subsequence_info& b) {
        bracket_subsequence_info res;
        int mn = min(a.open, b.close);
        res.ans = a.ans + b.ans + mn * 2;
        res.open = a.open + b.open - mn;
        res.close = a.close + b.close - mn;
        return res;
    }
};

struct good_index_info { // determine the minimum and maximum index where a good index is 
                         // a[0, i] all <= i and i <= a[i, n - 1]
                         // basically it's a index that left half is less than i and right half all greater than i
                         // for(int i = 0; i < n; i++) {
                         //     if(min(0, i) <= i && i <= max(i, n - 1)) {
                         //         return true
                         //     }
                         // }
    // https://www.codechef.com/problems/DOUBLEFLIPQ?tab=statement
    // update i with info(i + 1, a[i] == i ? i : -1)
    // then lazy segtree update(max(a[x], pos[a[x]])) to n - 1 with -1, the one that holds the mn == 0 is the good index we're looking for
    // careful cause [2, 1] doesn't have good index but it's the edge case of mn == 0 as well
    // update min_id and max_id each time the index changes

    int mn, min_id, max_id;
    good_index_info(int x = -1, int id = -1) : mn(x), max_id(id == -1 ? -inf : id), min_id(id == -1 ? inf : id) {} // id is -1 if a[i] != i and i if a[i] == i

    bool good() { // determine if this good_index_info is good enough
        return mn == 0 && min_id < inf && max_id >= 0;
    }

    friend good_index_info operator+(const good_index_info& a, const good_index_info& b) {
        if(a.mn == -1) return b;
        if(b.mn == -1) return a;
        good_index_info res;
        res.mn = min(a.min_id != inf ? a.mn : inf, b.min_id != inf ? b.mn : inf);
        if(a.mn == res.mn) res = a;
        if(b.mn == res.mn) {
            res.min_id = min(res.min_id, b.min_id);
            res.max_id = max(res.max_id, b.max_id);
        }
        return res;
    }
};

struct diameter_info {
    // https://codeforces.com/contest/1192/problem/B
    // find max diameter of the tree
    // careful with tin, tout set up for distinct subtree
    // max(d[i] + d[j] - 2 * d[lca(i, j)])
    ll diameter, plus_max, minus_max, left_mix, right_mix;
    // plus_max = d[i]
    // minus_max = - 2 * d[i]
    // left_mix = d[i] - 2 * d[j]
    // right_mix = -2 * d[i] + d[j]
    diameter_info() : diameter(0), plus_max(0), minus_max(0), left_mix(0), right_mix(0) {}

    diameter_info& operator+=(const ll v) {
        plus_max += v;
        minus_max -= 2 * v;
        left_mix -= v;
        right_mix -= v;
        return *this;
    }

    friend diameter_info operator+(const diameter_info& a, const diameter_info& b) {
        diameter_info res;
        res.plus_max = max(a.plus_max, b.plus_max);
        res.minus_max = max(a.minus_max, b.minus_max);
        res.left_mix = max({a.left_mix, b.left_mix, a.plus_max + b.minus_max});
        res.right_mix = max({b.right_mix, a.right_mix, b.plus_max + a.minus_max});
        res.diameter = max({a.diameter, b.diameter, a.left_mix + b.plus_max, a.plus_max + b.right_mix});
        return res;
    }
};

struct sorted_info {
    // https://codeforces.com/contest/1982/problem/F
    int mn, mx, R, L;
    bool sorted;
    
    sorted_info(int x = inf) : mn(x), mx(x == inf ? -inf : x), R(x), L(x), sorted(true) {}

    friend sorted_info operator+(const sorted_info& a, const sorted_info& b) {
        if(a.mx == -inf) return b;
        if(b.mx == -inf) return a;
        sorted_info res;  
        res.mn = min(a.mn, b.mn);
        res.mx = max(a.mx, b.mx);
        res.L = a.L;
        res.R = b.R;
        res.sorted = a.sorted && b.sorted && a.R <= b.L;
        return res;
    }
};

struct power_sum_info { // keep track of sum of a^5 segtree sum
    mint s1, s2, s3, s4, s5;
    power_sum_info(mint x = 0)
      : s1(x),
        s2(x * x),
        s3(x * x * x),
        s4(x * x * x * x),
        s5(x * x * x * x * x)
    {}

    friend power_sum_info operator+(const power_sum_info& a, const power_sum_info& b) {
        power_sum_info res;
        res.s1 = a.s1 + b.s1;
        res.s2 = a.s2 + b.s2;
        res.s3 = a.s3 + b.s3;
        res.s4 = a.s4 + b.s4;
        res.s5 = a.s5 + b.s5;
        return res;
    }

    void apply(mint v, int len) {
        mint v2 = v * v;
        mint v3 = v2 * v;
        mint v4 = v3 * v;
        mint v5 = v4 * v;
        // s5 = s5 + 5*s4*v + 10*s3*v^2 + 10*s2*v^3 + 5*s1*v^4 + len*v^5
        s5 = s5 + 5 * s4 * v + 10 * s3 * v2 + 10 * s2 * v3 + 5 * s1 * v4 + mint(len) * v5;
        // s4 = s4 + 4*s3*v + 6*s2*v^2 + 4*s1*v^3 + len*v^4
        s4 = s4 + 4 * s3 * v + 6 * s2 * v2 + 4 * s1 * v3 + mint(len) * v4;
        // s3 = s3 + 3*s2*v + 3*s1*v^2 + len*v^3
        s3 = s3 + 3 * s2 * v + 3 * s1 * v2 + mint(len) * v3;
        // s2 = s2 + 2*s1*v + len*v^2
        s2 = s2 + 2 * s1 * v + mint(len) * v2;
        // s1 = s1 + len*v
        s1 = s1 + mint(len) * v;
    }
};

struct bad_pair_info {
    // count number of [l, r] such that their [for(int i..) for(int j...) s += a[i] * a[j], s is odd]
    // we work on prefix and it's bad when prefix[r] - prefix[l - 1] % 4 == {2, 3}
    // when query, we do queries_range(l - 1, r) not normal [l, r] bc we're working with prefix
    // https://codeforces.com/group/o09Gu2FpOx/contest/541484/problem/K
    // when update a[i] to v, we update the prefix [i, n] with a[i] == 0 ? 1 : -1
    int dp[4];
    ll bad;
    bad_pair_info(int p = -1) : bad(0) {
        mset(dp, 0);
        if(p == -1) return;
        dp[p] = 1;
    }
    
    friend bad_pair_info operator+(const bad_pair_info& a, const bad_pair_info& b) {
        bad_pair_info res;
        res.bad = a.bad + b.bad;
        for(int i = 0; i < 4; i++) {
            res.dp[i] = a.dp[i] + b.dp[i];
            res.bad += (ll)a.dp[i] * (b.dp[(i + 2) % 4] + b.dp[(i + 3) % 4]);
        }
        return res;
    }

    void apply(int x) {
        x = ((x % 4) + 4) % 4;
        int now[4];
        for(int i = 0; i < 4; i++) {
            now[(i + x) % 4] = dp[i];
        }
        for(int i=  0; i < 4; i++) {
            dp[i] = now[i];
        }
    }
};

struct max_k_subarray_info {
    // max k non-overlapping subarray sum
    // https://codeforces.com/contest/280/problem/D
    const static int K = 20;
    ll L[K + 1], R[K + 1], LR[K + 1], best[K + 1];
    max_k_subarray_info(ll x = -INF) {
        for(int i = 0; i <= K; i++) {
            L[i] = R[i] = LR[i] = (i ? x : -INF);
            best[i] = i ? max(0LL, x) : 0;
        }
    }

    friend max_k_subarray_info operator+(const max_k_subarray_info& a, const max_k_subarray_info& b) {
        if(a.L[1] == -INF) return b;
        if(b.L[1] == -INF) return a;
        max_k_subarray_info res;
        for(int l = 0; l <= K; l++) {
            for(int r = 0; l + r <= K; r++) {
                int k = l + r;
                res.L[k] = max(res.L[k], a.L[l] + b.best[r]);
                res.R[k] = max(res.R[k], a.best[l] + b.R[r]);
                res.LR[k] = max(res.LR[k], a.L[l] + b.R[r]);
                res.best[k] = max(res.best[k], a.best[l] + b.best[r]);
				if(r + 1 <= K) {
                    res.best[k] = max(res.best[k], a.R[l] + b.L[r + 1]);
                    res.L[k] = max(res.L[k], a.LR[l] + b.L[r + 1]);
                    res.R[k] = max(res.R[k], a.R[l] + b.LR[r + 1]);
                    res.LR[k] = max(res.LR[k], a.LR[l] + b.LR[r + 1]);
                }
//                if(l + 1 <= K) {
//                    res.best[k] = max(res.best[k], a.R[l + 1] + b.L[r]);
//                    res.L[k] = max(res.L[k], a.LR[l + 1] + b.L[r]);
//                    res.R[k] = max(res.R[k], a.R[l + 1] + b.LR[r]);
//                    res.LR[k] = max(res.LR[k], a.LR[l + 1] + b.LR[r]);
//                }

            }
        }
        return res;
    }
};

struct binomial_info {
    // https://codeforces.com/problemset/problem/266/E
    mint s[6] = {};
    binomial_info() {}
    binomial_info(int x, int pos) {
        mint v = x;
        for(int p = 0; p < 6; p++)
            s[p] = v * mint(pos + 1).pow(p);
    }
    friend binomial_info operator+(binomial_info const &A, binomial_info const &B) {
        binomial_info R;
        for(int p = 0; p < 6; p++)
            R.s[p] = A.s[p] + B.s[p];
        return R;
    }
    void apply(int l, int r, mint x) {
        for(int p = 0; p < 6; p++) {
            mint sum_ip = pre[p][r + 1] - pre[p][l];
            s[p] = sum_ip * x;
        }
    }
    mint get_res(int l, int k) const {
        mint ans = 0;
        mint base = mint(-l);
        for(int p = 0; p <= k; p++)
            ans += comb.nCk(k, p) * base.pow(k - p) * s[p];
        return ans;
    }
};
