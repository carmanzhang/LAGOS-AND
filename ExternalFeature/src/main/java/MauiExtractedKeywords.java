import maui.main.MauiWrapper;

import java.io.*;
import java.util.ArrayList;

public class MauiExtractedKeywords {
    public static void main(String[] args) throws Exception {

//		String vocabularyName = "agrovoc_en";
        String vocabularyName = "mesh";
        String modelName = "nlm500";
        String dataDirectory = "Dependency-Feature/";

        MauiWrapper wrapper = new MauiWrapper(dataDirectory, vocabularyName, modelName);
        String filePath = "/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/resources/maui/SONG_GS_WHU_mix_dataset_pubmed_title_abstract.tsv";
        BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File("/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/resources/maui/SONG_GS_WHU_mix_dataset_pubmed_extracted_keywords.tsv")));
        String str;

        while((str = reader.readLine()) != null) {
            String[] splts = str.split("\t");
            String pm_ao = splts[0];
            String title_abstract = splts[1];
            if (title_abstract.length() < 30) {
                writer.write(pm_ao + "\t" + "" + "\n");
                continue;
            }
//			ArrayList<String> keywords = wrapper.extractTopicsFromFile(title_abstract, 15);
            ArrayList<String> keywords = wrapper.extractTopicsFromText(title_abstract, 15);
            String allKeywords = "";
            for (String keyword : keywords) {
                allKeywords += (keyword.toLowerCase() + "|");
            }
            writer.write(pm_ao + "\t" + allKeywords + "\n");
        }
    }
}
