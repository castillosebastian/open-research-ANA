import {
    CopilotRuntime,
    OpenAIAdapter,
    copilotRuntimeNextJSAppRouterEndpoint,
    langGraphPlatformEndpoint,
} from '@copilotkit/runtime';
import OpenAI from 'openai';
import { NextRequest } from 'next/server';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const serviceAdapter = new OpenAIAdapter({ openai });

// Add logging to debug deployment URL
const deploymentUrl = process.env.DEPLOYMENT === 'local' ? process.env.LOCAL_DEPLOYMENT_URL : process.env.DEPLOYMENT_URL;
console.log('Deployment URL:', deploymentUrl);
console.log('Deployment mode:', process.env.DEPLOYMENT);

if (!deploymentUrl) {
    throw new Error('Deployment URL is not configured properly');
}

const runtime = new CopilotRuntime({
    remoteEndpoints: [
        langGraphPlatformEndpoint({
            deploymentUrl: deploymentUrl,
            langsmithApiKey: process.env.LANGSMITH_API_KEY ?? '',
            agents: [{
                name: 'agent',
                description: 'Research assistant',
            }],
        })
    ]
});

export const POST = async (req: NextRequest) => {
    try {
        const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
            runtime,
            serviceAdapter,
            endpoint: '/api/copilotkit',
        });

        return handleRequest(req);
    } catch (error) {
        console.error('Error in copilotkit route:', error);
        return new Response(JSON.stringify({ error: 'Internal Server Error' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
        });
    }
};