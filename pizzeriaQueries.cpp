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
    vi root;
    SegmentTree(int n)
    {
        this->n = n;
        root.resize(n * 4);
    }

    void update(int index, int val)
    {
        update(0, 0, n - 1, index, val);
    }

    void update(int i, int left, int right, int index, int val)
    {
        if(left == right)
        {
            root[i] = val;
            return;
        }
        int middle = left + (right - left) / 2;
        if(index <= middle) update(i * 2 + 1, left, middle, index, val);
        else update(i * 2 + 2, middle + 1, right, index, val);
        root[i] = min(root[i * 2 + 1], root[i * 2 + 2]);
    }

    int get(int start, int end)
    {
        return get(0, 0, n - 1, start, end);
    }

    int get(int i, int left, int right, int start, int end)
    {
        if(left >= start && right <= end) return root[i];
        if(start > right || left > end) return INF;
        int middle = left + (right - left) / 2;
        return min(get(i * 2 + 1, left, middle, start, end), get(i * 2 + 2, middle + 1, right, start, end));
    }

};

void solve()
{
    int n, q;
    cin >> n >> q;
    SegmentTree up(n), down(n);
    for(int i = 0; i < n; i++)
    {
        int val;
        cin >> val;
        down.update(i, val - i);
        up.update(i, val + i);
    }

    while(q--)
    {
        int type;
        cin >> type;
        if(type == 1)
        {
            int k, x;
            cin >> k >> x;
            k--;
            down.update(k, x - k);
            up.update(k, x + k);
        }
        else
        {
            int k;
            cin >> k;
            k--;
            int left = down.get(0, k) + k;
            int right = up.get(k, n - 1) - k;
            cout << min(left, right) << endl;

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

