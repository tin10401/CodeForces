#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1LL << 61;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
 
int modExpo(int base, int exp, int mod)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}
 
class SegmentTree
{
    public:
    int n;
    vi root, lazy, lazy2; 
    SegmentTree(vi& arr)
    {
        this->n = arr.size();
        root.resize(n * 4), lazy.resize(n * 4), lazy2.resize(n * 4);
        build(0, 0, n - 1, arr);
    }
 
    void build(int i, int left, int right, vi& arr)
    {
        if(left == right)
        {
            root[i] = arr[left];
            return;
        }
        int middle = left + (right - left) / 2;
        build(i * 2 + 1, left, middle, arr);
        build(i * 2 + 2, middle + 1, right, arr);
        root[i] = root[i * 2 + 1] + root[i * 2 + 2];
    }
 
    void update(int start, int end, int x, bool add)
    {
        update(0, 0, n - 1, start, end, x, add);
    }
 
    void update(int i, int left, int right, int start, int end, int x, bool add)
    {
        apply(i, left, right);
        if(start > right || left > end) return;
        if(left >= start && right <= end)
        {
            if(add) lazy[i] += x;
            else lazy2[i] = x;
            apply(i, left, right);
            return;
        }
        int middle = left + (right - left) / 2;
        update(i * 2 + 1, left, middle, start, end, x, add);
        update(i * 2 + 2, middle + 1, right, start, end, x, add);
        root[i] = root[i * 2 + 1] + root[i * 2 + 2];
    }
    
    void apply(int i, int left, int right) // lazy1 for updating addition of x, lazy2 for setting all element equal to x
    {
        if(lazy2[i]) root[i] = (right - left + 1) * lazy2[i]; 
        else root[i] += (right - left + 1) * lazy[i];
        if(left != right)
        {
            if(lazy2[i])
            {
                lazy2[i * 2 + 1] = lazy2[i * 2 + 2] = lazy2[i];
            }
            else
            {
                if(lazy2[i * 2 + 1]) lazy2[i * 2 + 1] += lazy[i];
                else lazy[i * 2 + 1] += lazy[i];
                if(lazy2[i * 2 + 2]) lazy2[i * 2 + 2] += lazy[i];
                else lazy[i * 2 + 2] += lazy[i];
            }
        }
        lazy[i] = lazy2[i] = 0;
    }
     
    int get(int start, int end)
    {
        return get(0, 0, n - 1, start, end);
    }
 
    int get(int i, int left, int right, int start, int end)
    {
        apply(i, left, right);
        if(left >= start && right <= end) return root[i];
        if(left > end || start > right) return 0;
        int middle = left + (right - left) / 2;
        return get(i * 2 + 1, left, middle, start, end) + get(i * 2 + 2, middle + 1, right, start, end);
    }
};
 
void solve()
{
    int n, q;
    cin >> n >> q;
    vi arr(n);
    for(auto& it : arr) cin >> it;
    SegmentTree root(arr);
    while(q--)
    {
        int type;
        cin >> type;
        if(type != 3)
        {
            int a, b, c;
            cin >> a >> b >> c;
            a--, b--;
            root.update(a, b, c, type == 1);
        }
        else
        {
            int a, b;
            cin >> a >> b;
            a--, b--;
            cout << root.get(a, b) << endl;
        }
    }
}
 
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}
