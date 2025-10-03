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
const static int MX = 5e5 + 5;

const int BUF_SZ = 1 << 15; // do init_output() at the start of the main function

inline namespace Input {
    char buf[BUF_SZ];
    int pos;
    int len;
    char next_char() {
        if (pos == len) {
            pos = 0;
            len = (int)fread(buf, 1, BUF_SZ, stdin);
            if (!len) { return EOF; }
        }
        return buf[pos++];
    }

    int read_int() {
        int x;
        char ch;
        int sgn = 1;
        while (!isdigit(ch = next_char())) {
            if (ch == '-') { sgn *= -1; }
        }
        x = ch - '0';
        while (isdigit(ch = next_char())) { x = x * 10 + (ch - '0'); }
        return x * sgn;
    }
}
inline namespace Output {
    char buf[BUF_SZ];
    int pos;

    void flush_out() {
        fwrite(buf, 1, pos, stdout);
        pos = 0;
    }

    void write_char(char c) {
        if (pos == BUF_SZ) { flush_out(); }
        buf[pos++] = c;
    }

    void write_int(ll x) {
        static char num_buf[100];
        if (x < 0) {
            write_char('-');
            x *= -1;
        }
        int len = 0;
        for (; x >= 10; x /= 10) { num_buf[len++] = (char)('0' + (x % 10)); }
        write_char((char)('0' + x));
        while (len) { write_char(num_buf[--len]); }
        write_char('\n');
    }

    void init_output() { assert(atexit(flush_out) == 0); }
}

int n;
struct EulerianPath {
    int nodes, edges;
    bool directed;
    vvpii graph;
    vi deg, indeg, outdeg;
    vt<bool> used;
    vi ans_edges, ans_nodes;

    EulerianPath(int _nodes, bool _directed = false)
      : nodes(_nodes), edges(0), directed(_directed), graph(_nodes) {
        if(directed) indeg.assign(nodes,0), outdeg.assign(nodes,0);
        else deg.assign(nodes,0);
    }

    void add_edge(int u, int v, int id) {
        graph[u].emplace_back(v, id);
        edges++;
        if(directed) {
            outdeg[u]++;
            indeg[v]++;
        } else {
            graph[v].emplace_back(u, id);
            deg[u]++;
            deg[v]++;
        }
    }

    int find_start() const {
        int start = -1;
        if(!directed) {
            int odd = 0;
            for(int i = 0; i < nodes; i++) {
                if(deg[i] & 1) {
                    return -1;
                }
                if(start < 0 && deg[i] > 0) start = i;
            }
            if(start < 0) return 0;
        } else {
            int plus1 = 0, minus1 = 0;
            for(int i = 0; i < nodes; i++) {
                int d = outdeg[i] - indeg[i];
                if(d == 1) { plus1++; start = i; }
                else if(d == -1) minus1++;
                else if(d != 0) return -1;
                if(start < 0 && outdeg[i] > 0) start = i;
            }
            if(start < 0) return 0;
            if(!((plus1 == 1 && minus1 == 1) || (plus1 == 0 && minus1 == 0))) return -1;
        }
        return start;
    }

    int cnt = 0;
    void dfs(int u) {
		if(used.empty()) {
			used.rsz(edges);
		}
        while(!graph[u].empty()) {
            auto [v, id] = graph[u].back();
            graph[u].pop_back();
			while((int)used.size() <= id) used.pb(0);
            if(used[id]) continue;
            used[id] = true;
            dfs(v);
            cnt++;
        }
        if(u < 2 * n) {
            ans_nodes.pb(u);
        }
    }

    vi get_path() {
        int start = find_start();
        if(start < 0) return {};
        used.rsz(edges);
        dfs(start);
        if(cnt != edges) return {};
        rev(ans_nodes);
        return ans_nodes;
    }
};

const int K = 20;
int a[MX], b[MX];
void solve() {
    n = read_int();
    for(int i = 0; i < n; i++) {
        a[i] = read_int();
        b[i] = read_int();
    }
    auto f = [&](int i) -> vi {
        const int N = 1 << i;
        const int M = n * 2 + N;
        EulerianPath graph(M, false);
        int m = 0;
        for(int j = 0; j < n; j++) {
            int u = a[j] & ((1LL << i) - 1);
            int v = b[j] & ((1LL << i) - 1);
            graph.add_edge(2 * j, u + n * 2, m++);
            graph.add_edge(2 * j + 1, v + n * 2, m++);
            graph.add_edge(2 * j, 2 * j + 1, m++);
        }
        return graph.get_path();
    };
    int left = 1, right = 20;
    int res = 0;
    vi ans(2 * n);
    iota(all(ans), 0);
    while(left <= right) {
        int middle = (left + right) >> 1;
        auto nodes = f(middle);
        if(!nodes.empty()) swap(ans, nodes), res = middle, left = middle + 1;
        else right = middle - 1;
    }
    cout << res << '\n';
    vi seen(2 * n);
    for(auto& x : ans) {
        if(x < 2 * n) {
            if(!seen[x]) {
                cout << x + 1 << ' ';
                seen[x] = true;
            }
        }
    }
    cout << '\n';
}

signed main() {
    IOS;
    startClock
    init_output();
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
