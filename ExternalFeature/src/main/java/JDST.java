import gov.nih.nlm.nls.tc.Api.JdiApi;
import gov.nih.nlm.nls.tc.Api.StiApi;
import gov.nih.nlm.nls.tc.FilterApi.InputFilterOption;
import gov.nih.nlm.nls.tc.FilterApi.LegalWordsOption;
import gov.nih.nlm.nls.tc.FilterApi.OutputFilter;
import gov.nih.nlm.nls.tc.FilterApi.OutputFilterOption;
import gov.nih.nlm.nls.tc.Lib.Configuration;
import gov.nih.nlm.nls.tc.Lib.Count2f;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class JDST {

    public static void main(String[] args) throws Exception {
        String file = "/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/resources/pubmed_all_text_content.tsv";
        BufferedReader br = new BufferedReader(new FileReader(file));
        BufferedWriter fw = new BufferedWriter(new FileWriter("/home/zhangli/mydisk-2t/repo/pubmed-paper-author-link/resources/jd_st.tsv"));
        String line;
        String path_to_configuration = "/home/zhangli/mydisk-2t/repo/AuthorNameDisambiguation/test/retrievers_test/jnius/data/Config/tc.properties";
        JdiApi jdi = new JdiApi(path_to_configuration);
        StiApi sti = new StiApi(new Configuration(path_to_configuration, false));

        while ((line = br.readLine()) != null) {
            System.out.println("----------------------------------------------------------------------------");
            String[] split = line.split("\t");
            String pm_id = split[0];
            String text = split[1];
            String source = split[2];
            System.out.println(pm_id + "\t" + source);
            StringBuilder content = new StringBuilder();
            List<String> journal_descriptors = getJD(jdi, text);
            for (String journal_descriptor : journal_descriptors) {
                System.out.println(journal_descriptor);
                content.append(journal_descriptor).append("|");
            }
            String jd = content.toString();
            System.out.println();
            content = new StringBuilder();
            List<String> semantic_types = getST(sti, text);
            for (String semantic_type : semantic_types) {
                System.out.println(semantic_type);
                content.append(semantic_type).append("|");
            }
            String st = content.toString();
            fw.write(pm_id + "\t" + jd + "\t" + st + "\n");
        }
        br.close();
        fw.close();
    }

    private static List<String> getST(StiApi sti, String text) {
        Vector<Count2f> scores;
        OutputFilterOption output_filter_option;
        String[] result;

        scores = sti.GetStiScoresByText(text, new InputFilterOption(LegalWordsOption.DEFAULT_JDI));

        output_filter_option = new OutputFilterOption();
        output_filter_option.SetOutputNum(3);
        result = OutputFilter.ProcessText(scores, sti.GetSemanticTypes(), output_filter_option).split("\n");
        List<String> semantic_types = new ArrayList<String>();

        if (result.length > 5) {
            for (int i = 0; i < result.length; i++) {
                if (i == 2 || i == 3 || i == 4) {
                    String[] ttt = result[i].split("\\|");
                    String blyad = ttt[4].trim().replace(",", ".").toLowerCase();
                    semantic_types.add(blyad);
                }
            }
        }
        return semantic_types;
    }

    private static List<String> getJD(JdiApi jdi, String text) {
        Vector<Count2f> scores = jdi.GetJdiScoresByTextMesh(text, new InputFilterOption(LegalWordsOption.DEFAULT_JDI));

        OutputFilterOption output_filter_option = new OutputFilterOption();
        output_filter_option.SetOutputNum(3);

        String[] result = OutputFilter.ProcessText(scores, jdi.GetJournalDescriptors(), output_filter_option).split("\n");

        List<String> journal_descriptors = new ArrayList<String>();

        if (result.length > 5) {
            for (int i = 0; i < result.length; i++) {

                if (i == 2 || i == 3 || i == 4) {
                    String[] ttt = result[i].split("\\|");
                    String blyad = ttt[3].trim().replace(",", ".").toLowerCase();
                    journal_descriptors.add(blyad);
                }
            }
        }
        return journal_descriptors;
    }
}
