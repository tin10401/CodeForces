#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define vt vector
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ld long db
#define ll long long
#define ull unsigned long long
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vvpii vt<vpii>
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define vs vt<string>
#define pb push_back
#define ff first
#define ss second
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define SORTED(x) is_sorted(all(x))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#if defined(LOCAL) && __has_include("debug.h")
  #include "debug.h"
#else
  #define debug(...)
  #define startClock
  #define endClock
  inline void printMemoryUsage() {}
#endif
template<class T> using max_heap = priority_queue<T>; template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<typename T, size_t N> istream& operator>>(istream& is, array<T, N>& arr) { for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is; }
template<typename T, size_t N> istream& operator>>(istream& is, vector<array<T, N>>& vec) { for (auto &arr : vec) { is >> arr; } return is; }
template<typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) { return in >> input.ff >> input.ss; }
template<typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const static ll INF = 4e18 + 10;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

template<class T, typename F = function<T(const T&, const T&)>>
class basic_segtree {
public:
    int n;    
    int size;  
    vt<T> root;
    F func;
    T DEFAULT;  
    
    basic_segtree() {}

    basic_segtree(int _n, T _DEFAULT, F _func = [](const T& a, const T& b) {return a + b;}) : n(_n), func(_func), DEFAULT(_DEFAULT) {
        size = 1;
        while(size < _n) size <<= 1;
        root.assign(size << 1, _DEFAULT);
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
        bool has_left = false, has_right = false;
        while(l <= r) {
            if((l & 1) == 1) {
                if(!has_left) res_left = root[l++];
                else res_left = func(res_left, root[l++]); 
                has_left = true;
            }
            if((r & 1) == 0) {
                if(!has_right) res_right = root[r--];
                else res_right = func(root[r--], res_right);
                has_right = true;
            }
            l >>= 1; r >>= 1;
        }
        if(!has_left) return res_right;
        if(!has_right) return res_left;
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

struct info {
    struct part {
        ll s, mx, edge;
        int len;
    };

    vector<part> pre, suff;
    ll s = 0, mx = -1, pref = -1, suf = -1;
    int len = 0, best = 0;

    info() = default;

    info(ll x) {
        s = x; mx = x; pref = x; suf = x; len = 1; best = 0;
        pre.pb({x, x, -1, 1});
        suff.pb({x, x, -1, 1});
    }

    friend info operator+(const info& A, const info& B) {
        if(A.len == 0) return B;
        if(B.len == 0) return A;

        auto ok = [](const part& x) {
            return x.edge == -1 || x.edge >= x.s;
        };

        info C;
        C.best = max(A.best, B.best);
        C.s = A.s + B.s;
        C.len = A.len + B.len;
        C.pref = A.pref;
        C.suf  = B.suf;
        C.mx = max(A.mx, B.mx);
        C.pre = A.pre;
        if(!C.pre.empty() && C.pre.back().edge == -1) {
            C.pre.back().edge = B.pref;
            if(!ok(C.pre.back())) C.pre.pop_back();
        }
        for(auto x : B.pre) {
            x.mx = max(x.mx, A.mx);
            x.s += A.s;
            x.len += A.len;
            if(ok(x)) C.pre.pb(x);
        }

        C.suff = B.suff;
        if(!C.suff.empty() && C.suff.back().edge == -1) {
            C.suff.back().edge = A.suf;
            if(!ok(C.suff.back())) C.suff.pop_back();
        }
        for(auto x : A.suff) {
            x.mx = max(x.mx, B.mx);
            x.s += B.s;
            x.len += B.len;
            if(ok(x)) C.suff.push_back(x);
        }

        for(const auto& x : A.suff) {
            for(const auto& y : B.pre) {
                ll M = max(x.mx, y.mx);
                ll sm = x.s + y.s;
                if(M * 2 < sm) C.best = max(C.best, x.len + y.len);
            }
        }
        return C;
    }
};

void solve() {
    int n, m; cin >> n >> m;
    basic_segtree<info> root(n + 1, info(), [](const auto& x, const auto& y) {return x + y;});
    for(int i = 1; i <= n; i++) {
        ll x; cin >> x;
        root.update_at(i, info(x));
    }
    while(m--) {
        int op; cin >> op;
        if(op == 2) {
            ll i, x; cin >> i >> x;
            root.update_at(i, info(x));
            continue;
        }
        int l, r; cin >> l >> r;
        auto it = root.queries_range(l, r);
        auto ans = it.best;
        cout << (ans < 3 ? -1 : ans) << '\n';
    }
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
