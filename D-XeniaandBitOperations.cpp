// Author : Tin Le
//   __________________
//  | ________________ |
//  ||          ____  ||
//  ||   /\    |      ||
//  ||  /__\   |      ||
//  || /    \  |____  ||
//  ||________________||
//  |__________________|
//  \###################\
//   \###################\
//    \        ____       \
//     \_______\___\_______\
// An AC a day keeps the doctor away.

#pragma GCC optimize("Ofast")
#pragma GCC optimize ("unroll-loops")
#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ll unsigned long long
#define int long long
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define us unordered_set
#define um unordered_map
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define var(x) vector<ar(x)>
#define pq priority_queue
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define umii um<int, int, custom>
#define usi us<int, custom>
#define f first
#define s second
#define rsz resize
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#ifdef LOCAL
#define startClock clock_t tStart = clock();
#define endClock cout << fixed << setprecision(10) << "\nTime Taken: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
#else
#define startClock
#define endClock
#endif
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const static int INF = 1LL << 61;
const static int MX = 2e5 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }
const vvi dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr int modExpo(int base, int exp, int mod) { int res = 1; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
static uint64_t x;
uint64_t next() { uint64_t z = (x += 0x9e3779b97f4a7c15); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) * 0x94d049bb133111eb; return z ^ (z >> 31); }
struct custom{ template <typename T> size_t operator()(const T& value) const { return next() ^ std::hash<T>{}(value); } };

class SegmentTree
{
    public:
    int n, x;
    vi root;
    SegmentTree(vi& arr)
    {
        n = arr.size(); root.rsz(n * 4);
        x = log2(n);
        build(0, 0, n - 1, 0, arr);
    }

    void build(int i, int left, int right, int depth, vi& arr)
    {
        if(left == right) 
        {
            root[i] = arr[left];
            return;
        }
        int middle = left + (right - left) / 2;
        build(i * 2 + 1, left, middle, depth + 1, arr);
        build(i * 2 + 2, middle + 1, right, depth + 1, arr);
        merge(i, depth);
    }
    
    void update(int index, int val)
    {
        update(0, 0, n - 1, 0, index, val);
    }

    void update(int i, int left, int right, int depth, int index, int val)
    {
        if(left == right)
        {
            root[i] = val;
            return;
        }
        int middle = left + (right - left) / 2;
        if(index <= middle) update(i * 2 + 1, left, middle, depth + 1, index, val);
        else update(i * 2 + 2, middle + 1, right, depth + 1, index, val);
        merge(i, depth);
    }

    int get() {return root[0];}
    void merge(int i, int depth)
    {
        if(depth % 2 != x % 2) root[i] = root[i * 2 + 1] | root[i * 2 + 2];
        else root[i] = root[i * 2 + 1] ^ root[i * 2 + 2];
    }
};        

void solve()
{
    int n, q; cin >> n >> q;
    n = 1 << n;
    vi arr(n);
    for(auto& it : arr) cin >> it;
    SegmentTree root(arr);
    while(q--)
    {
        int a, b; cin >> a >> b;
        root.update(--a, b);
        cout << root.get() << endl;
    }
}

signed main()
{
    IOS;
    startClock

    int t = 1;
    // cin >> t;
    while(t--) solve();

    endClock
    return 0;
}

