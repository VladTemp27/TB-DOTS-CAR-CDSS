# UI Flowchart — TB-DOTS CAR CDSS Web App

Generated from `web-app/src/App.tsx` (routes) and per-page `navigate(...)` calls.

## Screen flow

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Inter, system-ui, sans-serif','fontSize':'14px','lineColor':'#94a3b8','primaryBorderColor':'#cbd5e1'},'flowchart':{'curve':'basis','nodeSpacing':55,'rankSpacing':70}}}%%
flowchart TD
    Login(["🔐 /login<br/><b>Login</b>"])

    subgraph Auth["🛡️ ProtectedRoute &nbsp;·&nbsp; redirects to /login if unauthenticated"]
        direction TB
        Home["🏠 /<br/><b>Home</b><br/><i>Patient Overview</i>"]
        Dashboard["📊 /dashboard<br/><b>Dashboard</b>"]

        subgraph Intake["📝 New Patient Intake · wizard"]
            direction LR
            Step1["/patient/new<br/><b>Step 1</b><br/><i>Demographics</i>"]
            Step2["/patient/new/lab<br/><b>Step 2</b><br/><i>Lab</i>"]
            Xray["/patient/new/xray<br/><b>Step 3</b><br/><i>X-ray Upload</i>"]
        end

        Result["🩺 /patient/:id/result<br/><b>Diagnostic Result</b>"]
        Features["🔬 /patient/:id/features<br/><b>Feature Contribution</b><br/><i>SHAP</i>"]
        Treatment["💊 /patient/:id/treatment<br/><b>Treatment Selection</b>"]
        Profile["👤 /patient/:id<br/><b>Patient Profile</b>"]
        Chart["📈 /patient/:id/chart<br/><b>Patient Chart</b>"]
        Checkin["🗓️ /patient/:id/checkin<br/><b>Monthly Check-in</b>"]
        RiskUpdate["⚠️ /patient/:id/risk-update<br/><b>Risk Update</b>"]
    end

    %% Auth
    Login ==>|login success| Home

    %% Persistent nav (sidebar / bottom nav)
    Home -. nav .-> Dashboard
    Home -. nav .-> Step1

    %% Intake wizard
    Home ==>|New patient| Step1
    Step1 ==>|Next| Step2
    Step2 ==>|Next| Xray
    Step2 -.->|Back| Step1
    Xray -.->|Back| Step2
    Xray ==>|Commit · run inference| Result

    %% Diagnostic result
    Result -->|View features| Features
    Result ==>|Proceed| Treatment

    %% Patient overview → chart
    Home ==>|Click patient card| Chart

    %% Patient Profile hub
    Profile --> Features
    Profile --> Checkin
    Profile --> Treatment
    Profile --> Chart
    Profile -.->|Back| Home

    %% Patient Chart
    Chart --> Checkin
    Chart --> Treatment
    Chart --> Profile
    Chart -.->|Back| Home

    %% Treatment → back to profile
    Treatment -->|Save / Back| Profile

    %% Monthly check-in → risk update
    Checkin ==>|Submit| RiskUpdate
    RiskUpdate --> Features
    RiskUpdate --> Chart
    RiskUpdate --> Profile

    %% Feature contribution → back to profile
    Features -.->|Back| Profile

    %% ---- styling ----
    classDef auth fill:#fee2e2,stroke:#ef4444,stroke-width:1.5px,color:#7f1d1d;
    classDef nav fill:#e0f2fe,stroke:#0ea5e9,stroke-width:1.5px,color:#075985;
    classDef intake fill:#fef9c3,stroke:#eab308,stroke-width:1.5px,color:#713f12;
    classDef diag fill:#dcfce7,stroke:#22c55e,stroke-width:1.5px,color:#14532d;
    classDef patient fill:#ede9fe,stroke:#8b5cf6,stroke-width:1.5px,color:#4c1d95;
    classDef followup fill:#ffedd5,stroke:#f97316,stroke-width:1.5px,color:#7c2d12;

    class Login auth;
    class Home,Dashboard nav;
    class Step1,Step2,Xray intake;
    class Result,Features diag;
    class Profile,Chart,Treatment patient;
    class Checkin,RiskUpdate followup;

    style Auth fill:#f8fafc,stroke:#cbd5e1,stroke-width:1px,color:#475569;
    style Intake fill:#fffbeb,stroke:#fcd34d,stroke-width:1px,color:#92400e;
```

**Legend** — 🔵 persistent nav · 🟡 intake wizard · 🟢 diagnostic · 🟣 patient hub · 🟠 follow-up loop.
Thick arrows (**⟹**) = primary path · thin = secondary action · dotted = back/nav.

## Persistent navigation

Available on every protected screen via the desktop sidebar (`DesktopLayout`) and the
mobile bottom bar (`BottomNav`):

| Label | Route |
|-------|-------|
| Patients / Patient Overview | `/` |
| Dashboard | `/dashboard` |
| New Patient | `/patient/new` |
| Logout | → `/login` |

## Two entry paths into a patient

1. **New patient (wizard):** Home → Intake Step 1 → Step 2 (Lab) → X-ray → **Diagnostic Result** → Treatment.
2. **Existing patient:** Home (patient card) → **Patient Chart** → Profile / Check-in / Treatment.

The **Patient Profile** acts as the central hub for an existing patient, linking out to
Features (SHAP), Monthly Check-in, Treatment, and Chart.
