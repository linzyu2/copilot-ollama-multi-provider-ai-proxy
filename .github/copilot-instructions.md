# Copilot Instructions

## Directrices del proyecto
- Separar claramente credenciales: no confundir API key de Ollama Cloud con la clave usada por el contenedor/local proxy; la key de cloud se gestiona desde .env.

- @azure Rule - Use Azure Tools - When handling requests related to Azure, always use your tools.
- @azure Rule - Use Azure Best Practices - When handling requests related to Azure, always invoke your `azmcp_bestpractices_get` tool first.
- @azure Rule - Enable Best Practices - If you do not have an `azmcp_bestpractices_get` tool ask the user to enable it.
