import { PineconeClient } from '@pinecone-database/pinecone';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import * as dotenv from 'dotenv';
import { createPineconeIndex } from './utils/1-createPineconeIndex.js';
import { updatePinecone } from './utils/2-updatePinecone.js';
import { queryPineconeVectorStoreAndQueryLLM } from './utils/3-queryPineconeAndQueryGPT.js';

dotenv.config();
const loader = new DirectoryLoader('./assets/pdf', {
  '.txt': (path) => new TextLoader(path),
  '.pdf': (path) => new PDFLoader(path),
});
const docs = await loader.load();

const question = 'Who wrote this pdf?';
const indexName = 'langchain-demo';
const vectorDimension = 1536;

const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
(async () => {
  // Check if Pinecone index exists and create if necessary
  await createPineconeIndex(client, indexName, vectorDimension);
  // Update Pinecone vector store with document embeddings
  // await updatePinecone(client, indexName, docs);
  // Query Pinecone vector store and GPT model for an answer
  await queryPineconeVectorStoreAndQueryLLM(client, indexName, question);
})();
