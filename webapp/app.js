const $=id=>document.getElementById(id);
const endpoint=$('endpoint'),status=$('status'),tools=$('tools'),output=$('output');
let sessionId=null;
async function rpc(method,params={},id=1){
  const headers={'Content-Type':'application/json','Accept':'application/json, text/event-stream'};
  if(sessionId) headers['Mcp-Session-Id']=sessionId;
  const r=await fetch(endpoint.value.trim(),{method:'POST',headers,body:JSON.stringify({jsonrpc:'2.0',id,method,params})});
  const sid=r.headers.get('Mcp-Session-Id'); if(sid) sessionId=sid;
  const text=await r.text();
  output.textContent=text||`HTTP ${r.status}`;
  if(!r.ok) throw new Error(`HTTP ${r.status}: ${text}`);
  return text;
}
$('connect').onclick=async()=>{
  status.textContent='🟡 Connecting...';tools.textContent='Loading tools...';sessionId=null;
  try{
    await rpc('initialize',{protocolVersion:'2025-06-18',capabilities:{},clientInfo:{name:'AlpacaMCPWeb',version:'1.0.0'}},1);
    await rpc('notifications/initialized',{},2);
    const raw=await rpc('tools/list',{},3);
    let parsed; try{parsed=JSON.parse(raw)}catch{}
    const list=parsed?.result?.tools;
    tools.textContent=list?.length?list.map(t=>`✓ ${t.name}`).join('\n'):raw;
    status.textContent='🟢 MCP Connected';
  }catch(e){status.textContent='🔴 Connection failed';tools.textContent=e.message}
};
