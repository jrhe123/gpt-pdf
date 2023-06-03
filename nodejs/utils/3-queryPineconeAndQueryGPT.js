import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import { loadQAStuffChain } from 'langchain/chains';
import { Document } from 'langchain/document';

export const queryPineconeVectorStoreAndQueryLLM = async (
  client,
  indexName,
  question,
) => {
  // Retrieve the Pinecone index
  const index = client.Index(indexName);
  // Create query embedding
  const queryEmbedding = await new OpenAIEmbeddings().embedQuery(question);
  // Query Pinecone index and return top 10 matches
  let queryResponse = await index.query({
    queryRequest: {
      topK: 10,
      vector: queryEmbedding,
      includeMetadata: true,
      includeValues: true,
    },
  });
  // Log the number of matches
  console.log(`Found ${queryResponse.matches.length} matches...`);
  // Log the question being asked
  console.log(`Asking question: ${question}...`);
  if (queryResponse.matches.length) {
    // Create an OpenAI instance and load the QAStuffChain
    const llm = new OpenAI({});
    const chain = loadQAStuffChain(llm);
    // Extract and concatenate page content from matched documents
    const concatenatedPageContent = queryResponse.matches
      .map((match) => match.metadata.pageContent)
      .join(' ');
    // Execute the chain with input documents and question
    const result = await chain.call({
      input_documents: [new Document({ pageContent: concatenatedPageContent })],
      question: question,
    });
    // Log the answer
    console.log(`Answer: ${result.text}`);
  } else {
    // Log that there are no matches, so GPT-3 will not be queried
    console.log('Since there are no matches, GPT-3 will not be queried.');
  }
};
