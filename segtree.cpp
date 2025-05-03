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
        root[i] = (ll)val * (right - left + 1);
        lazy[i] = val;
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
	
	T get() {
		return root[0];
	}
	
	template<typename Pred> // seg.min_left(ending, [](const int& a) {return a > 0;});
        int min_left(int ending, Pred f) { // min index where f[l, ending] is true
            return find_left(entireTree, ending, f);
        }

    template<typename Pred>
        int max_right(int starting, Pred f) {
            return find_right(entireTree, starting, f);
        }

    template<typename Pred>
        int find_left(iter, int end, Pred f) {
            pushDown;
            if (left > end) return -2;
            if (f(root[i])) return left;
            if (left == right) return -1;
            int middle = midPoint;
            int r = find_left(rp, end, f);
            if (r == -2) return find_left(lp, end, f);
            if (r == middle + 1) {
                int l = find_left(lp, end, f);
                if (l != -1) return l;
            }
            return r;
        }

    template<typename Pred>
        int find_right(iter, int start, Pred f) {
            pushDown;
            if (right < start) return -2;
            if (f(root[i])) return right;
            if (left == right) return -1;
            int middle = midPoint;
            int l = find_right(lp, start, f);
            if (l == -2) return find_right(rp, start, f);
            if (l == middle) {
                int r = find_right(rp, start, f);
                if (r != -1) return r;
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
        for (idx >>= 1; idx > 0; idx >>= 1) root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
    }
    
    T queries_range(int l, int r) {
        l = max(0, l), r = min(r, n - 1);
        T res_left = DEFAULT, res_right = DEFAULT;
        l += size, r += size;
        while (l <= r) {
            if ((l & 1) == 1) res_left = func(res_left, root[l++]);
            if ((r & 1) == 0) res_right = func(root[r--], res_right);
            l >>= 1; r >>= 1;
        }
        return func(res_left, res_right);
    }
	
	T queries_at(int idx) {
        if(idx <= 0 || idx >= n) return DEFAULT;
        return root[idx + size];
    }
	
	void update_range(int l, int r, ll v) {}

    T get() {
        return root[1];
    }
};

template<class T, class I = int, typename F = function<T(const T&, const T&)>>
class iterative_lazy_segtree {
public:
    int n, size, h;
    vt<T> seg;
    vt<I> lazy;
    vi L, R;
    F f;
    T default_val;
    
    iterative_lazy_segtree(int n, T default_val, F f)
        : n(n), default_val(default_val), f(f)
    {
        size = 1;
        while(size < n) size <<= 1;
        seg.assign(size << 1, default_val);
        lazy.assign(size << 1, 0);
        L.assign(size << 1, 0);
        R.assign(size << 1, 0);
        h = 0;
        for(int i = size; i > 0; i >>= 1)
            h++;
        for (int i = 0; i < size; i++) {
            L[size + i] = i;
            R[size + i] = i;
        }
        for (int i = size - 1; i > 0; i--) {
            L[i] = L[i << 1];
            R[i] = R[i << 1 | 1];
        }
    }
    
    T get() {
        return seg[1];
    }
    
    inline void update_at(int idx, T val) {
        if(idx < 0 || idx >= n) return;
        idx += size;
        push_to(idx);
        seg[idx] = val;
        rebuild_from(idx);
    }
    
    inline void update_range(int l, int r, I val) {
        if(l < 0 || r >= n || l > r) return;
        int Lq = l + size, Rq = r + size;
        push_to(Lq);
        push_to(Rq);
        int l0 = Lq, r0 = Rq;
        while(Lq <= Rq) {
            if(Lq & 1) { apply(Lq, val); Lq++; }
            if(!(Rq & 1)) { apply(Rq, val); Rq--; }
            Lq >>= 1; Rq >>= 1;
        }
        rebuild_from(l0);
        rebuild_from(r0);
    }
    
    inline T queries_range(int l, int r) {
        if(l < 0 || r >= n || l > r) return default_val;
        int Lq = l + size, Rq = r + size;
        push_to(Lq);
        push_to(Rq);
        T res_left = default_val, res_right = default_val;
        while(Lq <= Rq) {
            if(Lq & 1) res_left = f(res_left, seg[Lq++]);
            if(!(Rq & 1)) res_right = f(seg[Rq--], res_right);
            Lq >>= 1; Rq >>= 1;
        }
        return f(res_left, res_right);
    }
    
    inline T queries_at(int idx) {
        if(idx < 0 || idx >= n) return default_val;
        idx += size;
        push_to(idx);
        return seg[idx];
    }
    
private:
    inline void apply(int i, I val) {
        seg[i] += (ll) val * (R[i] - L[i] + 1);
        lazy[i] += val;
    }
    
    inline void push(int i) {
        if(lazy[i] != 0) {
            apply(i << 1, lazy[i]);
            apply(i << 1 | 1, lazy[i]);
            lazy[i] = 0;
        }
    }
    
    inline void push_to(int i) {
        for(int s = h; s >= 1; s--) {
            int idx = i >> s;
            push(idx);
        }
    }
    
    inline void rebuild_from(int i) {
        for(i /= 2; i > 0; i /= 2) {
            seg[i] = f(seg[i << 1], seg[i << 1 | 1]);
            if(lazy[i] != 0)
                seg[i] += (ll) lazy[i] * (R[i] - L[i] + 1);
        }
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

struct mex_tree {
    // change merge to min(left, right)
    // change the update to be root[curr] = delta;
    PSGT<int> seg;
    int n;

    mex_tree(const vector<int>& a, int starting_mex = 0) : n(a.size()) {
        seg.reset();
        seg.assign(n + 2, -1);
        int prev = 0;
        seg.add(0, prev, 0, starting_mex == 0 ? -1 : inf);
        for(int i = 1; i <= n + 1; i++) {
            seg.add(0, prev, i, -1);
        }
        for (int i = 0; i < n; ++i) {
            int v = min(a[i], n + 1);
            seg.add(i + 1, prev, v, i);
        }
    }

    int mex(int l, int r) const {
        return find_mex(t[r + 1], 0, n + 1, l);
    }

private:
    int find_mex(int curr, int L, int R, int bound) const {
        if (L == R) return L;
        int M = (L + R) >> 1;
        if (root[child[curr].ff] < bound) return find_mex(child[curr].ff, L, M, bound);
        return find_mex(child[curr].ss, M + 1, R, bound);
    }
};

// you have to set up by assigning size and updating from 0 to n - 1 first
const int MM = MX * 150;
int t[MX], ptr;
ll root[MM], lazy[MM];
pii child[MM];
template<typename T>
struct lazy_PSGT {
    int n;
    T DEFAULT;
    void assign(int n, T DEFAULT) {
        this->n = n;
        this->DEFAULT = DEFAULT;
    }

    T merge(T a, T b) {
        return a + b;
    }

    int create_node(int prev) {
        ++ptr;
        assert(ptr < MM);
        root[ptr] = root[prev];
        lazy[ptr] = lazy[prev];
        child[ptr] = child[prev];
        return ptr;
    }

    void apply(int curr, int left, int right, T val) {
        root[curr] += val * (right - left + 1);
        lazy[curr] += val;
    }

    void push_down(int curr, int left, int right) {
        if(lazy[curr] == 0 || left == right) return;
        int middle = midPoint;
        if(child[curr].ff) {
            child[curr].ff = create_node(child[curr].ff);
            apply(child[curr].ff, left, middle, lazy[curr]); 
        }
        if(child[curr].ss) {
            child[curr].ss = create_node(child[curr].ss);
            apply(child[curr].ss, middle + 1, right, lazy[curr]);
        }
        lazy[curr] = 0;
    }

    void update_range(int i, int prev, int start, int end, T delta) {
        update_range(t[i], prev, delta, start, end, 0, n - 1);
    }

    void update_range(int& curr, int prev, T delta, int start, int end, int left, int right) {
        push_down(curr, left, right);
        if(left > end || start > right) return;
        curr = create_node(prev);
        if(start <= left && right <= end) {
            apply(curr, left, right, delta);
            push_down(curr, left, right);
            return;
        }
        int middle = midPoint;
        update_range(child[curr].ff, child[prev].ff, delta, start, end, left, middle);
        update_range(child[curr].ss, child[prev].ss, delta, start, end, middle + 1, right);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

    T queries_range(int i, int start, int end) {
        return queries_range(t[i], start, end, 0, n - 1);
    }

    T queries_range(int curr, int start, int end, int left, int right) {
        push_down(curr, left, right);
        if(!curr || start > right || left > end) return DEFAULT;
        if(start <= left && right <= end) return root[curr];
        int middle = midPoint;
        return merge(queries_range(child[curr].ff, start, end, left, middle), queries_range(child[curr].ss, start, end, middle + 1, right));
    }

    int update_at(int i, int prev, int id, T delta) {
        update_at(t[i], prev, id, delta, 0, n - 1);
        return t[i];
    }

    void update_at(int &curr, int prev, int id, T delta, int left, int right) {  
        push_down(curr, left, right);
        curr = create_node(prev);
        if(left == right) { 
			root[curr] = merge(root[curr], delta);
            return;
        }
        int middle = midPoint;
        if(id <= middle) update_at(child[curr].ff, child[prev].ff, id, delta, left, middle); 
        else update_at(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

    void reset() {
        for(int i = 0; i <= ptr; i++) {
            root[i] = 0;
            child[i] = {0, 0};
            lazy[i] = 0;
        }
        for(int i = 0; i < min(ptr, (int)(sizeof(t)/sizeof(t[0]))); i++){
            t[i] = 0;
        }
        ptr = 0;
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
class implit_segtree {
    public:
    int n;
    implit_segtree(int n) {
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
    public:
    struct Node {
        T mx1, mx2, mn1, mn2, mx_cnt, mn_cnt, sm, ladd, lval;
        Node(T x = INF) : mx1(x), mx2(-INF), mn1(x), mn2(INF), mx_cnt(1), mn_cnt(1), sm(x), lval(INF), ladd(0) {}
    };
    int n;
    vt<Node> root;
    SGT_BEAT(int n) {
        this->n = n;
        root.rsz(n * 4);
    }

    void update_at(int id, T x) {
        update_at(entireTree, id, x);
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

    Node merge(const Node left, const Node right) {
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

    void update_min(int start, int end, T x) {
        update_min(entireTree, start, end, x);
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

    void update_max(int start, int end, T x) {
        update_max(entireTree, start, end, x);
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

    void update_val(int start, int end, T x) {
        update_val(entireTree, start, end, x);
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

    void update_add(int start, int end, T x) {
        update_add(entireTree, start, end, x);
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

    Node queries_range(int start, int end) {
        return queries_range(entireTree, start, end);
    }

    Node queries_range(iter, int start, int end) {
        pushDown;
        if(left > end || start > right) return Node();
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        return merge(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	Node queries_at(int id) {
		return queries_at(entireTree, id);
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

struct info {
    int x;
    info(int x = 0) : x(x) {}

    friend info operator+(const info& a, const info& b) {
        return info(a.x + b.x);
    }
};

